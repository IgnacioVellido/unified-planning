class DecompositionTree():
    """Stores de decomposition tree of a resulting plan"""

    def __init__(self, tree: str) -> None:
        self.tree = tree

    def __repr__(self) -> str:
        return self.tree