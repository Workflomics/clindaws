import unittest

import clingo

from clindaws.core.workflow import canonicalize_shown_symbols


def _symbol_strings(symbols: tuple[clingo.Symbol, ...]) -> tuple[str, ...]:
    return tuple(str(symbol) for symbol in symbols if symbol.name == "ape_bind")


class WorkflowCanonicalizationTests(unittest.TestCase):
    def test_same_signature_join_ports_are_canonicalized(self) -> None:
        signature = (("cat_a", ("value_a",)),)
        shown_symbols = (
            clingo.Function("tool_at_time", [clingo.Number(1), clingo.String("join")]),
            clingo.Function("ape_bind", [clingo.Number(1), clingo.Number(0), clingo.String("wf_input_b")]),
            clingo.Function("ape_bind", [clingo.Number(1), clingo.Number(1), clingo.String("wf_input_a")]),
        )

        canonical = canonicalize_shown_symbols(
            shown_symbols,
            {"join": (signature, signature)},
        )

        self.assertEqual(
            _symbol_strings(canonical),
            (
                'ape_bind(1,0,"wf_input_a")',
                'ape_bind(1,1,"wf_input_b")',
            ),
        )

    def test_paired_single_input_producers_follow_later_symmetric_join(self) -> None:
        signature = (("cat_a", ("value_a",)),)
        shown_symbols = (
            clingo.Function("tool_at_time", [clingo.Number(1), clingo.String("producer_a")]),
            clingo.Function("tool_at_time", [clingo.Number(2), clingo.String("producer_b")]),
            clingo.Function("tool_at_time", [clingo.Number(3), clingo.String("join")]),
            clingo.Function("ape_bind", [clingo.Number(1), clingo.Number(0), clingo.String("wf_input_b")]),
            clingo.Function("ape_bind", [clingo.Number(2), clingo.Number(0), clingo.String("wf_input_a")]),
            clingo.Function(
                "ape_bind",
                [
                    clingo.Number(3),
                    clingo.Number(0),
                    clingo.Function(
                        "out",
                        [clingo.Number(1), clingo.String("producer_a"), clingo.String("out_0")],
                    ),
                ],
            ),
            clingo.Function(
                "ape_bind",
                [
                    clingo.Number(3),
                    clingo.Number(1),
                    clingo.Function(
                        "out",
                        [clingo.Number(2), clingo.String("producer_b"), clingo.String("out_0")],
                    ),
                ],
            ),
            clingo.Function(
                "ape_holds_dim",
                [
                    clingo.Function(
                        "out",
                        [clingo.Number(1), clingo.String("producer_a"), clingo.String("out_0")],
                    ),
                    clingo.String("value_a"),
                    clingo.String("cat_a"),
                ],
            ),
            clingo.Function(
                "ape_holds_dim",
                [
                    clingo.Function(
                        "out",
                        [clingo.Number(2), clingo.String("producer_b"), clingo.String("out_0")],
                    ),
                    clingo.String("value_a"),
                    clingo.String("cat_a"),
                ],
            ),
        )

        canonical = canonicalize_shown_symbols(
            shown_symbols,
            {
                "producer_a": (signature,),
                "producer_b": (signature,),
                "join": (signature, signature),
            },
        )

        self.assertEqual(
            _symbol_strings(canonical),
            (
                'ape_bind(1,0,"wf_input_a")',
                'ape_bind(2,0,"wf_input_b")',
                'ape_bind(3,0,out(1,"producer_a","out_0"))',
                'ape_bind(3,1,out(2,"producer_b","out_0"))',
            ),
        )


if __name__ == "__main__":
    unittest.main()
