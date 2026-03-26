"""Brazilian calendar with national holidays, regional holidays, and commercial dates."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


def _easter_date(year: int) -> date:
    """Compute Easter Sunday using the Anonymous Gregorian algorithm.

    Args:
        year: Calendar year.

    Returns:
        Date of Easter Sunday for the given year.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_val = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_val) // 451
    month, day = divmod(h + l_val - 7 * m + 114, 31)
    return date(year, month, day + 1)


def get_national_holidays(year: int) -> dict[date, str]:
    """Get Brazilian national fixed and moveable holidays for a given year.

    Includes: Confraternizacao Universal, Carnaval (segunda/terca), Sexta-Feira Santa,
    Tiradentes, Dia do Trabalho, Corpus Christi, Independencia, Nossa Senhora Aparecida,
    Finados, Proclamacao da Republica, Consciencia Negra, Natal.

    Args:
        year: Calendar year.

    Returns:
        Dictionary mapping date to holiday name.
    """
    easter = _easter_date(year)

    holidays: dict[date, str] = {
        # Fixed national holidays
        date(year, 1, 1): "Confraternizacao Universal",
        date(year, 4, 21): "Tiradentes",
        date(year, 5, 1): "Dia do Trabalho",
        date(year, 9, 7): "Independencia do Brasil",
        date(year, 10, 12): "Nossa Senhora Aparecida",
        date(year, 11, 2): "Finados",
        date(year, 11, 15): "Proclamacao da Republica",
        date(year, 11, 20): "Dia da Consciencia Negra",
        date(year, 12, 25): "Natal",
        # Moveable holidays based on Easter
        easter - timedelta(days=48): "Carnaval (segunda-feira)",
        easter - timedelta(days=47): "Carnaval (terca-feira)",
        easter - timedelta(days=46): "Quarta-feira de Cinzas",
        easter - timedelta(days=2): "Sexta-Feira Santa",
        easter: "Pascoa",
        easter + timedelta(days=60): "Corpus Christi",
    }

    return holidays


def get_commercial_dates(year: int) -> dict[date, str]:
    """Get major Brazilian commercial dates that impact retail demand.

    Includes: Dia das Maes (2nd Sunday of May), Dia dos Namorados (June 12),
    Dia dos Pais (2nd Sunday of August), Dia das Criancas (Oct 12),
    Black Friday BR (4th Friday of November), Cyber Monday BR.

    Args:
        year: Calendar year.

    Returns:
        Dictionary mapping date to commercial event name.
    """
    commercial: dict[date, str] = {}

    # Dia das Maes - 2nd Sunday of May
    may_first = date(year, 5, 1)
    days_to_sunday = (6 - may_first.weekday()) % 7
    first_sunday_may = may_first + timedelta(days=days_to_sunday)
    mothers_day = first_sunday_may + timedelta(weeks=1)
    commercial[mothers_day] = "Dia das Maes"
    # Week before is high-demand period
    for i in range(1, 8):
        commercial[mothers_day - timedelta(days=i)] = "Semana Dia das Maes"

    # Dia dos Namorados - June 12
    commercial[date(year, 6, 12)] = "Dia dos Namorados"
    for i in range(1, 8):
        commercial[date(year, 6, 12) - timedelta(days=i)] = "Semana Dia dos Namorados"

    # Dia dos Pais - 2nd Sunday of August
    aug_first = date(year, 8, 1)
    days_to_sunday = (6 - aug_first.weekday()) % 7
    first_sunday_aug = aug_first + timedelta(days=days_to_sunday)
    fathers_day = first_sunday_aug + timedelta(weeks=1)
    commercial[fathers_day] = "Dia dos Pais"
    for i in range(1, 8):
        commercial[fathers_day - timedelta(days=i)] = "Semana Dia dos Pais"

    # Dia das Criancas - October 12
    commercial[date(year, 10, 12)] = "Dia das Criancas"

    # Black Friday BR - 4th Friday of November
    nov_first = date(year, 11, 1)
    days_to_friday = (4 - nov_first.weekday()) % 7
    first_friday_nov = nov_first + timedelta(days=days_to_friday)
    black_friday = first_friday_nov + timedelta(weeks=3)
    commercial[black_friday] = "Black Friday BR"
    commercial[black_friday + timedelta(days=2)] = "Cyber Monday BR"
    # Black Friday week
    for i in range(1, 5):
        commercial[black_friday - timedelta(days=i)] = "Black Week"

    # Natal shopping season (Dec 1-24)
    for day in range(1, 25):
        d = date(year, 12, day)
        if d not in commercial:
            commercial[d] = "Temporada Natal"

    return commercial


def get_regional_holidays(year: int, state: str = "SP") -> dict[date, str]:
    """Get regional holidays for a Brazilian state.

    Args:
        year: Calendar year.
        state: Two-letter state code (e.g., 'SP', 'RJ', 'MG').

    Returns:
        Dictionary mapping date to regional holiday name.
    """
    regional: dict[date, str] = {}

    state_holidays: dict[str, list[tuple[int, int, str]]] = {
        "SP": [(1, 25, "Aniversario de Sao Paulo"), (7, 9, "Revolucao Constitucionalista")],
        "RJ": [(4, 23, "Dia de Sao Jorge"), (11, 20, "Dia da Consciencia Negra")],
        "MG": [(4, 21, "Data Magna de Minas Gerais")],
        "BA": [(7, 2, "Independencia da Bahia")],
        "RS": [(9, 20, "Revolucao Farroupilha")],
        "PE": [(3, 6, "Revolucao Pernambucana")],
    }

    for month, day, name in state_holidays.get(state.upper(), []):
        regional[date(year, month, day)] = name

    return regional


def build_holiday_features(
    df: pd.DataFrame,
    date_col: str = "date",
    year: int | None = None,
    state: str = "SP",
    include_commercial: bool = True,
) -> pd.DataFrame:
    """Add holiday and commercial date features to a DataFrame.

    Adds columns: is_holiday, is_commercial_date, holiday_name, days_to_next_holiday,
    days_from_last_holiday.

    Args:
        df: Input DataFrame with a date column.
        date_col: Name of the date column.
        year: Year for holidays. If None, inferred from data.
        state: Brazilian state code for regional holidays.
        include_commercial: Whether to include commercial dates.

    Returns:
        DataFrame with holiday feature columns added.
    """
    df = df.copy()

    years = df[date_col].dt.year.unique() if year is None else [year]

    # Collect all holidays across years
    all_holidays: dict[date, str] = {}
    all_commercial: dict[date, str] = {}

    for y in years:
        all_holidays.update(get_national_holidays(y))
        all_holidays.update(get_regional_holidays(y, state=state))
        if include_commercial:
            all_commercial.update(get_commercial_dates(y))

    holiday_dates = set(all_holidays.keys())
    commercial_dates = set(all_commercial.keys())

    # Create feature columns
    df["is_holiday"] = df[date_col].dt.date.isin(holiday_dates).astype(int)
    df["is_commercial_date"] = df[date_col].dt.date.isin(commercial_dates).astype(int)

    # Holiday name (national holidays take priority)
    combined = {**all_commercial, **all_holidays}
    df["holiday_name"] = df[date_col].dt.date.map(combined).fillna("")

    # Days to next holiday / from last holiday
    sorted_holidays = sorted(holiday_dates | commercial_dates)
    if sorted_holidays:
        df["days_to_next_holiday"] = df[date_col].apply(
            lambda x: _days_to_next(x.date(), sorted_holidays)
        )
        df["days_from_last_holiday"] = df[date_col].apply(
            lambda x: _days_from_last(x.date(), sorted_holidays)
        )
    else:
        df["days_to_next_holiday"] = 0
        df["days_from_last_holiday"] = 0

    logger.info(
        "holiday_features_added",
        n_holidays=len(all_holidays),
        n_commercial=len(all_commercial),
        state=state,
    )
    return df


def _days_to_next(current: date, sorted_dates: list[date]) -> int:
    """Calculate days until the next event date."""
    for d in sorted_dates:
        if d >= current:
            return (d - current).days
    return 365


def _days_from_last(current: date, sorted_dates: list[date]) -> int:
    """Calculate days since the last event date."""
    for d in reversed(sorted_dates):
        if d <= current:
            return (current - d).days
    return 365
