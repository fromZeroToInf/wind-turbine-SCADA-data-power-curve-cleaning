import pandas as pd
import pandas.testing as pdt
from pathlib import Path

from wtpcc.pclib import pcf

def test_pc_filtering_keeps_points_inside_band() -> None:
    scadaData = pd.DataFrame(
        {
            "wind": [5.0, 6.0, 7.0],
            "power": [100, 250, 1000],
        }
    )
    
    powerCurve = pd.DataFrame(
        {
            "wind": [5.0, 6.0, 7.0],
            "power": [100, 250, 1000],
        }
    )
    
    filteredData, ram = pcf.pc_filtering(
        scadaData=scadaData,
        powerCurve=powerCurve,
        windSpec="wind",
        powerSpec="power",
        windowSize=1,
        powerMargin=0.0,
        minWindSpeed=4.0,
        measureRAM=False
    )
    
    assert ram is None
    pdt.assert_frame_equal(filteredData.reset_index(drop=True), scadaData)
    
def test_pc_filtering_keeps_points_inside_margin() -> None:
    scadaData = pd.DataFrame(
        {
            "wind": [5.0, 6.0, 7.0],
            "power": [100, 150, 1000],
        }
    )
    
    powerCurve = pd.DataFrame(
        {
            "wind": [5.0, 6.0, 7.0],
            "power": [100, 250, 1000],
        }
    )
    
    filteredData, ram = pcf.pc_filtering(
        scadaData=scadaData,
        powerCurve=powerCurve,
        windSpec="wind",
        powerSpec="power",
        windowSize=1,
        powerMargin=100.0,
        minWindSpeed=4.0,
        measureRAM=False
    )
    
    assert ram is None
    pdt.assert_frame_equal(filteredData.reset_index(drop=True), scadaData)
    
def test_pc_filtering_keeps_points_inside_margin2() -> None:
    scadaData = pd.DataFrame(
        {
            "wind": [5.0, 6.0, 7.0],
            "power": [100, 350, 1000],
        }
    )
    
    powerCurve = pd.DataFrame(
        {
            "wind": [5.0, 6.0, 7.0],
            "power": [100, 250, 1000],
        }
    )
    
    filteredData, ram = pcf.pc_filtering(
        scadaData=scadaData,
        powerCurve=powerCurve,
        windSpec="wind",
        powerSpec="power",
        windowSize=1,
        powerMargin=100.0,
        minWindSpeed=4.0,
        measureRAM=False
    )
    
    assert ram is None
    pdt.assert_frame_equal(filteredData.reset_index(drop=True), scadaData)

def test_pc_filtering_removes_points_outside_band() -> None:
    scadaData = pd.DataFrame(
        {
            "wind": [5.0, 6.0, 7.0],
            "power": [100.0, 999.0, 300.0],
        }
    )
    
    powerCurve = pd.DataFrame(
        {
            "wind": [5.0, 6.0, 7.0],
            "power": [100.0, 200.0, 300.0]
        }
    )
    
    filteredData, ram = pcf.pc_filtering(
        scadaData=scadaData,
        powerCurve=powerCurve,
        windSpec="wind",
        powerSpec="power",
        windowSize=1,
        powerMargin=0.0,
        minWindSpeed=4.0,
        measureRAM=False
    )
    
    sol = pd.DataFrame(
        {
            "wind": [5.0, 7.0],
            "power": [100.0, 300.0]
        }
    )
    
    assert ram is None
    assert len(filteredData) == 2
    pdt.assert_frame_equal(filteredData.reset_index(drop=True), sol)
    
def test_pc_filtering_points_transition_outside_band() -> None:
    scadaData = pd.DataFrame(
        {
            "wind": [5.0, 6.0, 7.0, 8.0, 9.0, 25.0],
            "power": [100.0, 200.0, 999.0, 400.0, 500.0, 100.0],
        }
    )
    
    powerCurve = pd.DataFrame(
        {
            "wind": [5.0, 6.0, 7.0, 8.0, 9.0, 25.0],
            "power": [100.0, 200.0, 300.0, 400.0, 500.0, 2000.0],
        }
    )
    
    filteredData, ram = pcf.pc_filtering(
        scadaData=scadaData,
        powerCurve=powerCurve,
        windSpec="wind",
        powerSpec="power",
        windowSize=1,
        powerMargin=0.0,
        minWindSpeed=4.0,
        measureRAM=False
    )
    
    sol = pd.DataFrame(
        {
            "wind": [5.0, 6.0, 8.0, 9.0],
            "power": [100.0, 200.0, 400.0, 500.0],
        }
    )
    assert ram is None
    pdt.assert_frame_equal(filteredData.reset_index(drop=True), sol)
    

def test_pc_filtering_respects_min_wind_speed() -> None:
    scadaData = pd.DataFrame(
        {
            "wind": [3.0, 4.0, 7.0, 8.0, 9.0, 25.0],
            "power": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        }
    )
    
    powerCurve = pd.DataFrame(
        {
            "wind": [3.0, 4.0, 7.0, 8.0, 9.0, 25.0],
            "power": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        }
    )
    filteredData, ram = pcf.pc_filtering(
        scadaData=scadaData,
        powerCurve=powerCurve,
        windSpec="wind",
        powerSpec="power",
        windowSize=1,
        powerMargin=0.0,
        minWindSpeed=4.0,
        measureRAM=False
    )
    sol = pd.DataFrame(
        {
            "wind": [7.0, 8.0, 9.0, 25.0],
            "power": [300.0, 400.0, 500.0, 600.0],
        }
    )
    assert ram is None
    pdt.assert_frame_equal(filteredData.reset_index(drop=True), sol)
    
def test_pc_filtering_returns_ram_value() -> None:
    scadaData = pd.DataFrame(
        {
            "wind": [3.0, 4.0, 7.0, 8.0, 9.0, 25.0],
            "power": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        }
    )
    
    powerCurve = pd.DataFrame(
        {
            "wind": [3.0, 4.0, 7.0, 8.0, 9.0, 25.0],
            "power": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
        }
    )
    filteredData, ram = pcf.pc_filtering(
        scadaData=scadaData,
        powerCurve=powerCurve,
        windSpec="wind",
        powerSpec="power",
        windowSize=1,
        powerMargin=0.0,
        minWindSpeed=4.0,
        measureRAM=True
    )
    sol = pd.DataFrame(
        {
            "wind": [7.0, 8.0, 9.0, 25.0],
            "power": [300.0, 400.0, 500.0, 600.0],
        }
    )
    assert isinstance(ram, int)
    assert ram > 0
    pdt.assert_frame_equal(filteredData.reset_index(drop=True), sol)
    
def test_pc_writes_single_csv_file(tmp_path) -> None:
    inputDir = tmp_path / "input"
    outDir = tmp_path / "output"
    inputDir.mkdir()
    outDir.mkdir()
    
    scadaData = pd.DataFrame(
        {
            "wind": [5.0, 6.0, 7.0],
            "power": [100, 250, 1000],
        }
    )
    scadaData.to_csv(inputDir / "wt1.csv", index=False)
    
    powerCurve = pd.DataFrame(
        {
            "wind": [5.0, 6.0, 7.0],
            "power": [100, 250, 1000],
        }
    )
    
    pcf.pc(
        inputDir=str(inputDir),
        outDir=str(outDir),
        powerCurve=powerCurve,
        windSpec="wind",
        powerSpec="power",
        windowSize=1,
        powerMargin=0.0,
        minWindSpeed=4.0,
        nJobs=1,
    )
    outPath = outDir / "pc_filtered_wt1.csv"
    assert outPath.exists()
    result = pd.read_csv(outPath)
    pdt.assert_frame_equal(result, scadaData)

def test_pc_writes_single_parquet_file(tmp_path) -> None:
    inputDir = tmp_path / "input"
    outDir = tmp_path / "output"
    inputDir.mkdir()
    outDir.mkdir()
    
    scadaData = pd.DataFrame(
        {
            "wind": [5.0, 6.0, 7.0],
            "power": [100, 250, 1000],
        }
    )
    scadaData.to_parquet(inputDir / "wt1.parquet", index=False)
    
    powerCurve = pd.DataFrame(
        {
            "wind": [5.0, 6.0, 7.0],
            "power": [100, 250, 1000],
        }
    )
    
    pcf.pc(
        inputDir=str(inputDir),
        outDir=str(outDir),
        powerCurve=powerCurve,
        windSpec="wind",
        powerSpec="power",
        windowSize=1,
        powerMargin=0.0,
        minWindSpeed=4.0,
        nJobs=1,
    )
    outPath = outDir / "pc_filtered_wt1.parquet"
    assert outPath.exists()
    result = pd.read_parquet(outPath)
    pdt.assert_frame_equal(result, scadaData)