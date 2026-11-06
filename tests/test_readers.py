"""Os formatos além do Parquet na comparação (`readers`, `io.open_source`): CSV, NDJSON, JSON, Arrow IPC, Avro, Excel,
SAS, Stata e SPSS, convertidos para Parquet numa pasta temporária que some no fim."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

import nini_datacomp as dc
from nini_datacomp import api
from nini_datacomp.io import open_source
from nini_datacomp.readers import ReadOptions, format_of

STAT = Path(__file__).parent / "fixtures" / "stat"

DF = pl.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "valor": [1.5, None, -3.25, 10.0],
        "nome": ["Ana", "Bia", None, "Zoë"],
    }
)


def _write(df: pl.DataFrame, path: Path) -> Path:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.write_csv(path)
    elif suffix == ".tsv":
        df.write_csv(path, separator="\t")
    elif suffix in (".ndjson", ".jsonl"):
        df.write_ndjson(path)
    elif suffix == ".json":
        df.write_json(path)
    elif suffix in (".arrow", ".ipc", ".feather"):
        df.write_ipc(path)
    elif suffix == ".avro":
        df.write_avro(path)
    else:
        df.write_parquet(path)
    return path


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".ndjson", ".jsonl", ".json", ".arrow", ".feather", ".avro"])
def test_formats_match_parquet(tmp_path: Path, suffix: str) -> None:
    other = _write(DF, tmp_path / f"dados{suffix}")
    base = _write(DF, tmp_path / "dados.parquet")
    res = dc.compare(other, base, "id")
    assert res.is_match, (res.errors, res.columns.type_mismatches)
    assert res.rows.common == 4


def test_staged_folder_is_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv = _write(DF, tmp_path / "dados.csv")
    (tmp_path / "tmp").mkdir()
    src = open_source(csv, ReadOptions(temp_dir=tmp_path / "tmp"), stage_bytes=1024)
    assert src.staged is not None
    staged_dir = src.staged.path
    assert staged_dir.is_dir()
    assert all(f.suffix == ".parquet" for f in src.files)
    assert src.lazy.collect().equals(DF)
    src.cleanup()
    assert not staged_dir.exists()
    # na comparação, a pasta some no fim, também quando dá erro
    tmp = tmp_path / "tmp2"
    tmp.mkdir()
    dc.compare(csv, DF, "id", read=ReadOptions(temp_dir=tmp))
    assert list(tmp.iterdir()) == []

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("falhou no meio")

    monkeypatch.setattr(api, "_compare", boom)
    with pytest.raises(RuntimeError, match="falhou no meio"):
        dc.compare(csv, DF, "id", read=ReadOptions(temp_dir=tmp))
    assert list(tmp.iterdir()) == []


def test_many_files_and_chunks(tmp_path: Path) -> None:
    """Uma pasta de CSV vira uma fonte só (os arquivos ocultos ficam de fora), comparada em várias janelas."""
    big = pl.DataFrame({"id": range(3000), "v": [i * 0.5 for i in range(3000)], "s": [f"t{i}" for i in range(3000)]})
    folder = tmp_path / "csv"
    folder.mkdir()
    for i in range(3):
        big.slice(i * 1000, 1000).write_csv(folder / f"parte{i}.csv")
    (folder / "_SUCCESS").write_text("", encoding="utf-8")
    src = open_source(folder)
    assert src.lazy.collect().sort("id").equals(big)
    res = dc.compare(folder, big, "id", window_rows=700)
    assert res.is_match
    assert res.execution.windows > 1


def test_folder_rules(tmp_path: Path) -> None:
    _write(DF, tmp_path / "a.parquet")
    _write(DF, tmp_path / "solto.csv")
    assert [f.name for f in open_source(tmp_path).files] == ["a.parquet"]  # pasta de Parquet: o CSV solto fica fora
    only_csv = open_source(tmp_path, ReadOptions(format="csv"))
    assert only_csv.lazy.collect().equals(DF)
    mixed = tmp_path / "misturada"
    mixed.mkdir()
    _write(DF, mixed / "a.csv")
    _write(DF, mixed / "b.ndjson")
    with pytest.raises(ValueError, match="mistura formatos"):
        open_source(mixed)
    flat = tmp_path / "rasa"
    (flat / "sub").mkdir(parents=True)
    _write(DF, flat / "sub" / "a.csv")
    with pytest.raises(FileNotFoundError):
        open_source(flat, ReadOptions(recursive=False))


def test_csv_options(tmp_path: Path) -> None:
    path = tmp_path / "br.txt"
    path.write_text("id;valor;nome\n1;1,5;Ana\n2;NA;Bia\n3;-3,25;\n4;10;Zoë\n", encoding="utf-8")
    opts = ReadOptions(format="csv", separator=";", decimal_comma=True, null_values=["NA", ""])
    res = dc.compare(path, DF, "id", read=opts)
    assert res.is_match, res.errors
    # latin1: convertido para UTF-8 antes de o Polars ler
    latin = tmp_path / "latin1.csv"
    latin.write_bytes("id,nome\n1,José\n2,São Paulo\n".encode("latin-1"))
    src = open_source(latin, ReadOptions(encoding="latin1"))  # a fonte guarda a pasta temporária: viva até o collect
    df = src.lazy.collect()
    assert df["nome"].to_list() == ["José", "São Paulo"]
    with pytest.raises(ValueError, match="só vale para SAS"):
        open_source(latin, ReadOptions(encoding="wlatin1"))
    # tipos de algumas colunas no lugar dos deduzidos
    typed_src = open_source(latin, ReadOptions(encoding="latin1", schema_overrides={"id": pl.Int8}))
    typed = typed_src.lazy.collect()
    assert typed.schema["id"] == pl.Int8


def test_read_options_per_side(tmp_path: Path) -> None:
    left = tmp_path / "l.csv"
    left.write_text("id;v\n1;2\n", encoding="utf-8")
    right = tmp_path / "r.csv"
    right.write_text("id|v\n1|2\n", encoding="utf-8")
    res = dc.compare(left, right, "id", read=(ReadOptions(separator=";"), ReadOptions(separator="|")))
    assert res.is_match


def test_read_options_validation() -> None:
    with pytest.raises(ValidationError):
        ReadOptions(separator=";;")
    with pytest.raises(ValidationError):
        ReadOptions(encoding="klingon")
    with pytest.raises(ValidationError):
        ReadOptions(nao_existe=1)  # type: ignore[call-arg]
    assert ReadOptions(encoding="wlatin1").encoding == "wlatin1"  # nome do SAS: vale para SAS/Stata/SPSS
    with pytest.raises(ValueError, match="formato desconhecido"):
        format_of(Path("x.abc"), ReadOptions())
    assert format_of(Path("x.abc"), ReadOptions(format="csv")) == "csv"
    assert format_of(Path("X.SAS7BDAT"), ReadOptions()) == "sas"


@pytest.mark.skipif(importlib.util.find_spec("fastexcel") is None, reason="ler Excel precisa do fastexcel")
def test_excel(tmp_path: Path) -> None:
    if importlib.util.find_spec("xlsxwriter") is None:
        pytest.skip("gravar o .xlsx do teste precisa do xlsxwriter")
    path = tmp_path / "dados.xlsx"
    DF.write_excel(path, worksheet="dados")
    assert dc.compare(path, DF, "id", read=ReadOptions(sheet="dados")).is_match


# ----------------------------------------------------------------------------- SAS, Stata e SPSS na comparação
@pytest.mark.parametrize("name", ["sample.sas7bdat", "sample.xpt", "sample.sav", "sample.zsav", "sample.dta"])
def test_compare_stat_files(tmp_path: Path, name: str) -> None:
    """O arquivo contra ele mesmo em Parquet (como a migração SAS -> data lake): igual; com um valor mudado, a
    divergência aparece na coluna certa."""
    path = STAT / name
    src = open_source(path)
    df = src.lazy.collect()
    src.cleanup()
    lake = tmp_path / "lake.parquet"
    df.write_parquet(lake)
    res = dc.compare(path, lake, "id")
    assert res.is_match, res.errors
    changed = df.with_columns(valor=pl.when(pl.col("id") == 3).then(-3.0).otherwise(pl.col("valor")))
    changed.write_parquet(lake)
    res = dc.compare(path, lake, "id")
    assert res.rows.mismatched == 1
    assert res.columns["valor"].mismatches == 1
    assert res.columns["nome"].mismatches == 0


def test_compare_sas_against_typed_lake(tmp_path: Path) -> None:
    """No SAS todo número é Float64; no lake o id costuma ser inteiro: ``cast`` resolve a chave."""
    lake = dc.read_sas(STAT / "sample.sas7bdat").with_columns(pl.col("id").cast(pl.Int64))
    res = dc.compare(STAT / "sample.sas7bdat", lake, "id")
    assert res.aborted
    assert "different types" in res.errors[0]
    assert dc.compare(STAT / "sample.sas7bdat", lake, "id", cast="right").is_match


def test_compare_real_sas_and_options(tmp_path: Path) -> None:
    """Texto vazio: nulo de um lado e "" do outro diverge, até a leitura usar a mesma regra."""
    path = tmp_path / "lake.parquet"
    dc.read_sas(STAT / "real.sas7bdat", blank_strings_as_null=False).write_parquet(path)
    assert not dc.compare(STAT / "real.sas7bdat", path, "column5").is_match
    same = dc.compare(STAT / "real.sas7bdat", path, "column5", read=ReadOptions(blank_strings_as_null=False))
    assert same.is_match, same.errors
    assert same.rows.common == 10


def test_compare_latin1_xpt() -> None:
    """O .xpt em Latin-1 na comparação, com ``encoding`` só no lado SAS."""
    other = pl.DataFrame({"id": [1.0, 2.0, 3.0], "nome": ["José", "São Paulo", "açú"]})
    res = dc.compare(STAT / "latin1.xpt", other, "id", read=(ReadOptions(encoding="latin1"), ReadOptions()))
    assert res.is_match, res.errors


def test_read_options_rejects_reader() -> None:
    with pytest.raises(ValidationError):
        ReadOptions(reader="native")  # type: ignore[call-arg]
