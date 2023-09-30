from opvious.common import (
    capturing_instance,
    method_decorator,
    to_camel_case,
    with_instance,
)


class TestCommon:
    def test_to_camel_case(self):
        assert to_camel_case("") == ""
        assert to_camel_case("foo") == "foo"
        assert to_camel_case("foo_bar") == "fooBar"
        assert to_camel_case("foo_bar_baz") == "fooBarBaz"

    def test_capturing_instance(self):
        @capturing_instance
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
            @capturing_instance
            def wrapper(fn):
                def wrapped(*args):
                    return f * fn(*args)

                return wrapped

            return wrapper

        class Foo:
            offset = 2
            factor = 3

            @with_instance(lambda self: multiply(self.factor))
            def plus_offset(self, n):
                return self.offset + n

        foo = Foo()

        assert foo.plus_offset(5) == 21

        fn = foo.plus_offset
        assert fn(7) == 27

    def test_method_decorator(self):
        @method_decorator()
        def multiply(f):
            def wrapper(meth):
                def wrapped(*args):
                    return f * meth(*args)

                return wrapped

            return wrapper

        class Foo:
            offset = 2
            factor = 3

            @multiply(lambda init, self: init(self.factor))
            def plus_offset(self, n):
                return self.offset + n

            @multiply(5)
            def minus_offset(self, n):
                return n - self.offset

        foo = Foo()

        assert foo.plus_offset(5) == 21
        assert foo.minus_offset(3) == 5

        plus = foo.plus_offset
        assert plus(7) == 27
        minus = foo.minus_offset
        assert minus(2) == 0
