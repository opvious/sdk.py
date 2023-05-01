from opvious.common import to_camel_case


class TestCommon:
    def test_to_camel_case(self):
        assert to_camel_case("") == ""
        assert to_camel_case("foo") == "foo"
        assert to_camel_case("foo_bar") == "fooBar"
        assert to_camel_case("foo_bar_baz") == "fooBarBaz"
