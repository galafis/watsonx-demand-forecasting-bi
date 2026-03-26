"""Tests for Brazilian calendar: Easter computation, national holidays, commercial dates."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.data.calendar_br import (
    _easter_date,
    build_holiday_features,
    get_commercial_dates,
    get_national_holidays,
    get_regional_holidays,
)


# ---------------------------------------------------------------------------
# Easter computation
# ---------------------------------------------------------------------------

class TestEasterDate:
    """Tests for the Anonymous Gregorian Easter algorithm."""

    def test_easter_2024(self) -> None:
        assert _easter_date(2024) == date(2024, 3, 31)

    def test_easter_2025(self) -> None:
        assert _easter_date(2025) == date(2025, 4, 20)

    def test_easter_2026(self) -> None:
        assert _easter_date(2026) == date(2026, 4, 5)

    def test_easter_2023(self) -> None:
        assert _easter_date(2023) == date(2023, 4, 9)

    def test_easter_2022(self) -> None:
        assert _easter_date(2022) == date(2022, 4, 17)

    def test_easter_always_in_march_or_april(self) -> None:
        for year in range(2000, 2050):
            easter = _easter_date(year)
            assert easter.month in (3, 4), f"Easter {year} not in March/April: {easter}"

    def test_easter_always_sunday(self) -> None:
        for year in range(2020, 2030):
            easter = _easter_date(year)
            assert easter.weekday() == 6, f"Easter {year} is not Sunday: {easter}"


# ---------------------------------------------------------------------------
# National holidays
# ---------------------------------------------------------------------------

class TestGetNationalHolidays:
    """Tests for Brazilian national holidays."""

    def test_fixed_holidays_present(self) -> None:
        holidays = get_national_holidays(2024)
        fixed = {
            date(2024, 1, 1): "Confraternizacao Universal",
            date(2024, 4, 21): "Tiradentes",
            date(2024, 5, 1): "Dia do Trabalho",
            date(2024, 9, 7): "Independencia do Brasil",
            date(2024, 10, 12): "Nossa Senhora Aparecida",
            date(2024, 11, 2): "Finados",
            date(2024, 11, 15): "Proclamacao da Republica",
            date(2024, 11, 20): "Dia da Consciencia Negra",
            date(2024, 12, 25): "Natal",
        }
        for d, name in fixed.items():
            assert d in holidays, f"Missing fixed holiday: {name}"
            assert holidays[d] == name

    def test_carnival_dates_2024(self) -> None:
        holidays = get_national_holidays(2024)
        easter = date(2024, 3, 31)
        carnival_monday = easter - timedelta(days=48)
        carnival_tuesday = easter - timedelta(days=47)
        assert carnival_monday in holidays
        assert carnival_tuesday in holidays
        assert "Carnaval" in holidays[carnival_monday]
        assert "Carnaval" in holidays[carnival_tuesday]

    def test_carnival_2025(self) -> None:
        holidays = get_national_holidays(2025)
        easter = date(2025, 4, 20)
        carnival_monday = easter - timedelta(days=48)
        carnival_tuesday = easter - timedelta(days=47)
        assert carnival_monday == date(2025, 3, 3)
        assert carnival_tuesday == date(2025, 3, 4)
        assert carnival_monday in holidays
        assert carnival_tuesday in holidays

    def test_good_friday(self) -> None:
        holidays = get_national_holidays(2024)
        easter = date(2024, 3, 31)
        good_friday = easter - timedelta(days=2)
        assert good_friday in holidays
        assert holidays[good_friday] == "Sexta-Feira Santa"

    def test_corpus_christi(self) -> None:
        holidays = get_national_holidays(2024)
        easter = date(2024, 3, 31)
        corpus_christi = easter + timedelta(days=60)
        assert corpus_christi in holidays
        assert holidays[corpus_christi] == "Corpus Christi"

    def test_ash_wednesday(self) -> None:
        holidays = get_national_holidays(2024)
        easter = date(2024, 3, 31)
        ash_wednesday = easter - timedelta(days=46)
        assert ash_wednesday in holidays
        assert holidays[ash_wednesday] == "Quarta-feira de Cinzas"

    def test_easter_in_holidays(self) -> None:
        holidays = get_national_holidays(2024)
        assert date(2024, 3, 31) in holidays
        assert holidays[date(2024, 3, 31)] == "Pascoa"

    def test_total_holiday_count(self) -> None:
        holidays = get_national_holidays(2024)
        # 9 fixed + 6 moveable (Carnival Mon, Carnival Tue, Ash Wed, Good Friday, Easter, Corpus Christi)
        assert len(holidays) == 15

    def test_different_years_different_moveable_dates(self) -> None:
        h2024 = get_national_holidays(2024)
        h2025 = get_national_holidays(2025)
        # Carnival dates should differ between years
        carnival_dates_2024 = [d for d, n in h2024.items() if "Carnaval" in n]
        carnival_dates_2025 = [d for d, n in h2025.items() if "Carnaval" in n]
        assert set(carnival_dates_2024) != set(carnival_dates_2025)


# ---------------------------------------------------------------------------
# Commercial dates
# ---------------------------------------------------------------------------

class TestGetCommercialDates:
    """Tests for Brazilian commercial dates."""

    def test_mothers_day_is_second_sunday_of_may(self) -> None:
        commercial = get_commercial_dates(2024)
        mothers_days = [d for d, n in commercial.items() if n == "Dia das Maes"]
        assert len(mothers_days) == 1
        md = mothers_days[0]
        assert md.month == 5
        assert md.weekday() == 6  # Sunday

    def test_valentines_day_brazil(self) -> None:
        commercial = get_commercial_dates(2024)
        assert date(2024, 6, 12) in commercial
        assert commercial[date(2024, 6, 12)] == "Dia dos Namorados"

    def test_fathers_day_is_second_sunday_of_august(self) -> None:
        commercial = get_commercial_dates(2024)
        fathers_days = [d for d, n in commercial.items() if n == "Dia dos Pais"]
        assert len(fathers_days) == 1
        fd = fathers_days[0]
        assert fd.month == 8
        assert fd.weekday() == 6  # Sunday

    def test_childrens_day(self) -> None:
        commercial = get_commercial_dates(2024)
        assert date(2024, 10, 12) in commercial
        assert commercial[date(2024, 10, 12)] == "Dia das Criancas"

    def test_black_friday_is_fourth_friday_of_november(self) -> None:
        commercial = get_commercial_dates(2024)
        bf_dates = [d for d, n in commercial.items() if n == "Black Friday BR"]
        assert len(bf_dates) == 1
        bf = bf_dates[0]
        assert bf.month == 11
        assert bf.weekday() == 4  # Friday

    def test_cyber_monday_after_black_friday(self) -> None:
        commercial = get_commercial_dates(2024)
        bf = [d for d, n in commercial.items() if n == "Black Friday BR"][0]
        cm = [d for d, n in commercial.items() if n == "Cyber Monday BR"][0]
        assert cm == bf + timedelta(days=2)

    def test_christmas_season_december(self) -> None:
        commercial = get_commercial_dates(2024)
        natal_dates = [d for d, n in commercial.items() if n == "Temporada Natal"]
        # Most of December 1-24 should be marked
        assert len(natal_dates) > 0
        for d in natal_dates:
            assert d.month == 12
            assert d.day < 25

    def test_mothers_day_week_before(self) -> None:
        commercial = get_commercial_dates(2024)
        week_dates = [d for d, n in commercial.items() if n == "Semana Dia das Maes"]
        assert len(week_dates) == 7  # 7 days before


# ---------------------------------------------------------------------------
# Regional holidays
# ---------------------------------------------------------------------------

class TestGetRegionalHolidays:
    """Tests for regional/state holidays."""

    def test_sp_holidays(self) -> None:
        regional = get_regional_holidays(2024, state="SP")
        assert date(2024, 1, 25) in regional  # Aniversario de SP
        assert date(2024, 7, 9) in regional   # Revolucao Constitucionalista

    def test_rj_holidays(self) -> None:
        regional = get_regional_holidays(2024, state="RJ")
        assert date(2024, 4, 23) in regional  # Dia de Sao Jorge

    def test_unknown_state_empty(self) -> None:
        regional = get_regional_holidays(2024, state="XX")
        assert len(regional) == 0

    def test_case_insensitive(self) -> None:
        regional = get_regional_holidays(2024, state="sp")
        assert len(regional) > 0


# ---------------------------------------------------------------------------
# Build holiday features
# ---------------------------------------------------------------------------

class TestBuildHolidayFeatures:
    """Tests for building holiday feature columns on a DataFrame."""

    @pytest.fixture
    def year_df(self) -> pd.DataFrame:
        """Full year DataFrame for 2024."""
        dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        return pd.DataFrame({"date": dates, "quantity": 100})

    def test_columns_created(self, year_df: pd.DataFrame) -> None:
        result = build_holiday_features(year_df)
        expected = ["is_holiday", "is_commercial_date", "holiday_name",
                    "days_to_next_holiday", "days_from_last_holiday"]
        for col in expected:
            assert col in result.columns

    def test_is_holiday_flags(self, year_df: pd.DataFrame) -> None:
        result = build_holiday_features(year_df)
        # Jan 1 should be a holiday
        jan1 = result[result["date"] == "2024-01-01"]
        assert jan1["is_holiday"].iloc[0] == 1

    def test_christmas_is_holiday(self, year_df: pd.DataFrame) -> None:
        result = build_holiday_features(year_df)
        xmas = result[result["date"] == "2024-12-25"]
        assert xmas["is_holiday"].iloc[0] == 1

    def test_regular_day_not_holiday(self, year_df: pd.DataFrame) -> None:
        result = build_holiday_features(year_df)
        # Pick a day unlikely to be any holiday
        jan3 = result[result["date"] == "2024-01-03"]
        assert jan3["is_holiday"].iloc[0] == 0

    def test_days_to_next_holiday_non_negative(self, year_df: pd.DataFrame) -> None:
        result = build_holiday_features(year_df)
        assert (result["days_to_next_holiday"] >= 0).all()

    def test_days_from_last_holiday_non_negative(self, year_df: pd.DataFrame) -> None:
        result = build_holiday_features(year_df)
        assert (result["days_from_last_holiday"] >= 0).all()

    def test_holiday_name_populated(self, year_df: pd.DataFrame) -> None:
        result = build_holiday_features(year_df)
        jan1 = result[result["date"] == "2024-01-01"]
        assert jan1["holiday_name"].iloc[0] == "Confraternizacao Universal"

    def test_commercial_date_flag(self, year_df: pd.DataFrame) -> None:
        result = build_holiday_features(year_df, include_commercial=True)
        # Dia dos Namorados (June 12) should be commercial
        june12 = result[result["date"] == "2024-06-12"]
        assert june12["is_commercial_date"].iloc[0] == 1

    def test_without_commercial_dates(self, year_df: pd.DataFrame) -> None:
        result = build_holiday_features(year_df, include_commercial=False)
        assert "is_commercial_date" in result.columns
        # June 12 should NOT be commercial when disabled
        june12 = result[result["date"] == "2024-06-12"]
        assert june12["is_commercial_date"].iloc[0] == 0
