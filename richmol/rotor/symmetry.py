class C1:
    @staticmethod
    def irrep_from_k_tau(k: int, tau: int) -> str:
        return "A"


class C2v(C1):
    @staticmethod
    def irrep_from_k_tau(k: int, tau: int) -> str:
        assert (tau in (0,1)), f"tau"
        return {(0, 0): "A1", (0, 1): "B1", (1, 0): "B2", (1, 1): "A2"}[(k%2, tau)]