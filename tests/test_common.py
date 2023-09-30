from opvious.common import method_decorator, to_camel_case, with_instance


class TestCommon:
    def test_to_camel_case(self):
        assert to_camel_case("") == ""
        assert to_camel_case("foo") == "foo"
        assert to_camel_case("foo_bar") == "fooBar"
        assert to_camel_case("foo_bar_baz") == "fooBarBaz"

    def test_method_decorator(self):
        @method_decorator
        def double(fn):
            def wrapped(*args):
                return 2 * fn(*args)

            return wrapped

        class Foo:
            offset = 1

            @double
            def plus_offset(self, n):
                return n + self.offset

            @property
            @double
            def two(self):
                return self.offset

        foo = Foo()

        # Direct calls
        assert foo.plus_offset(10) == 22
        assert foo.two == 2

        # Accessed as properties
        fn = foo.plus_offset
        assert fn(20) == 42

    def test_with_instance(self):
        def multiply(f):
            @method_decorator
            def wrapper(fn):
                def wrapped(*args):
                    return 2 * fn(*args)

                return wrapped

            return wrapper

        class Foo:
            offset = 2
            factor = 3

            @with_instance(lambda self: multiply(self.factor))
            def plus_offset(self, n):
                return self.offset + n

        foo = Foo()

        assert foo.plus_offset(5) == 14

        fn = foo.plus_offset
        assert fn(7) == 18
