from collections.abc import Callable


class ModuleRegistry[T]:
    def __init__(self) -> None:
        self._registry: dict[str, Callable[[], type[T]]] = {}
        self._labels: dict[str, str] = {}

    def register(self, name: str, label: str | None = None) -> Callable[[Callable[[], type[T]]], Callable[[], type[T]]]:
        def wrapper(loader: Callable[[], type[T]]) -> Callable[[], type[T]]:
            assert name not in self._registry, f"'{name}' already registered!"
            self._registry[name] = loader
            self._labels[name] = label or name
            return loader

        return wrapper

    def get(self, name: str) -> type[T]:
        if name not in self._registry:
            raise KeyError(f"'{name}' not registered. Available: {list(self._registry.keys())}")
        return self._registry[name]()

    def list_available(self) -> list[str]:
        return list(self._registry.keys())

    def labels(self) -> dict[str, str]:
        """Display label per registered name, in registration order."""
        return dict(self._labels)
