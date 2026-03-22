import pytest
from solution import parse_duration, format_duration


class TestParseDuration:
    """Tests for the parse_duration function."""

    # Happy path tests
    def test_parse_seconds_only(self):
        assert parse_duration("45s") == 45

    def test_parse_minutes_only(self):
        assert parse_duration("5m") == 300

    def test_parse_hours_only(self):
        assert parse_duration("2h") == 7200

    def test_parse_days_only(self):
        assert parse_duration("3d") == 259200

    def test_parse_weeks_only(self):
        assert parse_duration("1w") == 604800

    def test_parse_hours_and_minutes(self):
        assert parse_duration("2h30m") == 9000

    def test_parse_days_and_hours(self):
        assert parse_duration("1d12h") == 129600

    def test_parse_weeks_and_days(self):
        assert parse_duration("1w2d") == 777600

    def test_parse_full_combination(self):
        assert parse_duration("1w2d3h4m5s") == 788645

    def test_parse_hours_minutes_seconds(self):
        assert parse_duration("1h30m45s") == 5445

    def test_parse_single_second(self):
        assert parse_duration("1s") == 1

    def test_parse_single_minute(self):
        assert parse_duration("1m") == 60

    def test_parse_large_value(self):
        assert parse_duration("100h") == 360000

    def test_parse_large_week_count(self):
        assert parse_duration("52w") == 52 * 7 * 24 * 3600

    def test_parse_minutes_and_seconds(self):
        assert parse_duration("10m30s") == 630

    def test_parse_weeks_days_hours_minutes(self):
        assert parse_duration("1w1d1h1m") == 604800 + 86400 + 3600 + 60

    def test_parse_zero_seconds(self):
        assert parse_duration("0s") == 0

    def test_parse_zero_minutes(self):
        assert parse_duration("0m") == 0

    def test_parse_zero_hours(self):
        assert parse_duration("0h") == 0

    def test_parse_zero_weeks(self):
        assert parse_duration("0w") == 0

    def test_parse_multi_digit_values(self):
        assert parse_duration("10h45m") == 38700

    # Error cases
    def test_parse_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_duration("")

    def test_parse_none_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_duration(None)

    def test_parse_integer_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_duration(123)

    def test_parse_invalid_unit_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_duration("5x")

    def test_parse_plain_number_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_duration("100")

    def test_parse_negative_value_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_duration("-5h")

    def test_parse_float_value_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_duration("2.5h")

    def test_parse_unit_without_value_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_duration("h")

    def test_parse_spaces_raise_value_error(self):
        with pytest.raises(ValueError):
            parse_duration("2h 30m")

    def test_parse_leading_space_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_duration(" 2h")

    def test_parse_trailing_space_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_duration("2h ")

    def test_parse_uppercase_unit_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_duration("2H")

    def test_parse_invalid_characters_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_duration("2h@30m")

    def test_parse_list_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_duration(["2h"])

    def test_parse_only_unit_letters_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_duration("wdhms")

    def test_parse_special_characters_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_duration("!@#$")


class TestFormatDuration:
    """Tests for the format_duration function."""

    # Happy path tests
    def test_format_zero_seconds(self):
        assert format_duration(0) == "0s"

    def test_format_one_second(self):
        assert format_duration(1) == "1s"

    def test_format_seconds_only(self):
        assert format_duration(45) == "45s"

    def test_format_one_minute_exactly(self):
        assert format_duration(60) == "1m"

    def test_format_minutes_only(self):
        assert format_duration(300) == "5m"

    def test_format_one_hour_exactly(self):
        assert format_duration(3600) == "1h"

    def test_format_hours_only(self):
        assert format_duration(7200) == "2h"

    def test_format_one_day_exactly(self):
        assert format_duration(86400) == "1d"

    def test_format_days_only(self):
        assert format_duration(259200) == "3d"

    def test_format_one_week_exactly(self):
        assert format_duration(604800) == "1w"

    def test_format_hours_and_minutes(self):
        assert format_duration(9000) == "2h30m"

    def test_format_days_and_hours(self):
        assert format_duration(129600) == "1d12h"

    def test_format_weeks_and_days(self):
        assert format_duration(777600) == "1w2d"

    def test_format_full_combination(self):
        assert format_duration(788645) == "1w2d3h4m5s"

    def test_format_hours_minutes_seconds(self):
        assert format_duration(5445) == "1h30m45s"

    def test_format_minutes_and_seconds(self):
        assert format_duration(630) == "10m30s"

    def test_format_large_number_of_weeks(self):
        assert format_duration(52 * 7 * 24 * 3600) == "52w"

    def test_format_complex_duration(self):
        assert format_duration(604800 + 86400 + 3600 + 60) == "1w1d1h1m"

    def test_format_only_seconds_when_less_than_minute(self):
        assert format_duration(59) == "59s"

    def test_format_weeks_days_hours_minutes_seconds(self):
        total = 7 * 86400 + 2 * 86400 + 5 * 3600 + 20 * 60 + 10
        assert format_duration(total) == "1w2d5h20m10s"

    # Edge cases
    def test_format_59_seconds(self):
        assert format_duration(59) == "59s"

    def test_format_61_seconds(self):
        assert format_duration(61) == "1m1s"

    def test_format_3599_seconds(self):
        assert format_duration(3599) == "59m59s"

    def test_format_3601_seconds(self):
        assert format_duration(3601) == "1h1s"

    def test_format_86399_seconds(self):
        assert format_duration(86399) == "23h59m59s"

    def test_format_86401_seconds(self):
        assert format_duration(86401) == "1d1s"

    def test_format_604799_seconds(self):
        assert format_duration(604799) == "6d23h59m59s"

    def test_format_604801_seconds(self):
        assert format_duration(604801) == "1w1s"

    # Error cases
    def test_format_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            format_duration(-1)

    def test_format_large_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            format_duration(-1000)


class TestRoundTrip:
    """Tests that verify parse_duration and format_duration are inverses of each other."""

    def test_roundtrip_simple_seconds(self):
        assert parse_duration(format_duration(45)) == 45

    def test_roundtrip_simple_minutes(self):
        assert parse_duration(format_duration(300)) == 300

    def test_roundtrip_simple_hours(self):
        assert parse_duration(format_duration(7200)) == 7200

    def test_roundtrip_complex_duration(self):
        assert parse_duration(format_duration(788645)) == 788645

    def test_roundtrip_zero(self):
        assert parse_duration(format_duration(0)) == 0

    def test_roundtrip_one_week(self):
        assert parse_duration(format_duration(604800)) == 604800

    def test_roundtrip_full_combination(self):
        original = "1w2d3h4m5s"
        seconds = parse_duration(original)
        result = format_duration(seconds)
        assert result == original

    def test_roundtrip_hours_and_minutes(self):
        original = "2h30m"
        seconds = parse_duration(original)
        result = format_duration(seconds)
        assert result == original

    def test_format_then_parse_large_value(self):
        seconds = 1000000
        formatted = format_duration(seconds)
        assert parse_duration(formatted) == seconds

    def test_parse_then_format_preserves_value(self):
        duration_str = "1w3d7h25m50s"
        seconds = parse_duration(duration_str)
        assert format_duration(seconds) == duration_str
