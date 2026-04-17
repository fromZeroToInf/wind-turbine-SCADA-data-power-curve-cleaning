from pathlib import Path
from glob import glob
import pandas as pd
import numpy as np
import psutil
import gc
import os
from joblib import Parallel, delayed
from tqdm import tqdm

class pcf:
    
    @staticmethod
    def pc_filtering(
        scadaData: pd.DataFrame,
        powerCurve: pd.DataFrame,
        windSpec: str,
        powerSpec: str,
        windowSize: int,
        powerMargin: float = 150.0,
        minWindSpeed: float = 5.0,
        measureRAM: bool = False,
    ) -> tuple[pd.DataFrame, int | None]:
        
        windVals = scadaData[windSpec].to_numpy()
        powerVals = scadaData[powerSpec].to_numpy()
        
        pcWind = powerCurve[windSpec].to_numpy()
        pcPower = powerCurve[powerSpec].to_numpy()
        
        pc = np.interp(x=windVals,
                    xp=pcWind,
                    fp=pcPower)
        
        pcRange = ( windVals >= pcWind[0]) & (windVals <= pcWind[-1])
        
        minWind = windVals > minWindSpeed
        
        outOfBand = ((np.abs(powerVals - pc) > powerMargin) & pcRange & minWind).astype(np.int8)
        
        _filter = np.ones(windowSize, dtype=int)
        
        transitions = np.convolve(outOfBand, _filter, mode="same")
        
        mask = pcRange & minWind & (transitions == 0)
        
        filteredData = scadaData.loc[mask].copy()
        
        if measureRAM:
            ramUsage = psutil.virtual_memory().available
            return filteredData, ramUsage
        
        return filteredData, None

    @staticmethod
    def _read_file(path: Path) -> pd.DataFrame:
        _format = path.suffix.lower()
        
        if _format == ".csv":
            return pd.read_csv(path)
        if _format == ".parquet":
            return pd.read_parquet(path)
        
        raise ValueError(f"Unknown file type: {path}")
    
    @staticmethod
    def _write_file(df: pd.DataFrame, path: Path) -> None:
        _format = path.suffix.lower()
        csv = ".csv"
        parquet = ".parquet"

        if _format == csv:
            return df.to_csv(path, index=False)
        if _format == parquet:
            return df.to_parquet(path, index=False)
        
        raise ValueError(f"Unknown file type: {path}"
                        f"Known file types {[csv,parquet]}.")
    
    @staticmethod
    def _process_file(
        filePath: Path,
        outDir: Path,
        powerCurve: pd.DataFrame,
        windSpec: str,
        powerSpec: str,
        windowSize: int,
        powerMargin: float = 150.0,
        minWindSpeed: float = 5.0,
        measureRAM: bool = False,
    ) -> int | None:
        
        scadaData = pcf._read_file(filePath)
        
        filteredData, ram = pcf.pc_filtering(
            scadaData=scadaData,
            powerCurve=powerCurve,
            windSpec=windSpec,
            powerSpec=powerSpec,
            windowSize=windowSize,
            powerMargin=powerMargin,
            minWindSpeed=minWindSpeed,
            measureRAM=measureRAM,
        )
        
        pcf._write_file(filteredData, outDir / ("pc_filtered_"+filePath.name))
        
        del scadaData
        del filteredData
        gc.collect()
        
        return ram

    @staticmethod
    def _estimate_jobs(ramBefore: int, ramAfter: int) -> int:
        cpuCnt = max((os.cpu_count() or 1) -1, 1)
        
        ramUsed = max(ramBefore - ramAfter, 1)
        avail_ram = psutil.virtual_memory().available
        
        jobs = max(1, avail_ram // ramUsed)
        
        return max(1, min(cpuCnt, jobs))


    @staticmethod
    def pc(
        inputDir: str,
        outDir: str,
        powerCurve: pd.DataFrame,
        windSpec: str,
        powerSpec: str,
        windowSize: int,
        powerMargin: float = 150,
        minWindSpeed: float = 5.0,
        nJobs: int | None = None,
    ) -> None:
        """Applies power-curve filtering. Writes csv/parquet files to outDir.

        Args:
            inputDir (str): Make sure only relevant files are present.
            outDir (str): Directory to save files.
            powerCurve (pd.DataFrame): This table must contain the cols windSpec and powerSpec.
            windSpec (str): Column name used in powerCurve and files in inputDir.
            powerSpec (str): Column name used in powerCurve and files in inputDir
            windowSize (int): Data points may be in transition to outside of the powerMargin
            powerMargin (float, optional): Define upper and lower bound of the power curve. Defaults to 150 (kW).
            minWindSpeed (float, optional): Minimum windspeed for the filter. Defaults to 5.0.
            nJobs (int | None, optional): If None, nJobs are estimated by cpu count and RAM usage. Defaults to None.
        """
        inputDir = Path(inputDir)
        outDir = Path(outDir)
        outDir.mkdir(parents=True, exist_ok=True)
        
        files = list(inputDir.glob("*.*"))
        
        if not files:
            raise ValueError(f"No files found  in {inputDir}")
        
        if nJobs is None:
            ramBefore = psutil.virtual_memory().available
        
            ramAfter = pcf._process_file(
                filePath=files[0],
                outDir=outDir,
                powerCurve=powerCurve,
                windSpec=windSpec,
                powerSpec=powerSpec,
                windowSize=windowSize,
                powerMargin=powerMargin,
                minWindSpeed=minWindSpeed,
                measureRAM=True
            )
            if len(files) == 1:
                return
            
            jobs = pcf._estimate_jobs(ramBefore, ramAfter)
            
        Parallel(n_jobs=nJobs if nJobs is not None else jobs, backend="loky", verbose=10)(
            delayed(pcf._process_file)(
                filePath=fp,
                outDir=outDir,
                powerCurve=powerCurve,
                windSpec=windSpec,
                powerSpec=powerSpec,
                windowSize=windowSize,
                powerMargin=powerMargin,
                minWindSpeed=minWindSpeed,
            )
            for fp in tqdm(files[1:] if nJobs is None else files)
        )

    