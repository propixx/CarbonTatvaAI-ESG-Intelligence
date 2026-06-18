import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "training"))

from build_kpi_summary_dataset import (
    build_metadata,
    build_target,
    clean_kpis,
    compute_yoy,
    make_example,
    SourceRow,
)


class KpiSummaryDatasetTests(unittest.TestCase):
    def test_yoy_uses_reduction_sign_convention(self):
        self.assertAlmostEqual(compute_yoy(80, 100), 20)
        self.assertAlmostEqual(compute_yoy(120, 100), -20)

    def test_negative_current_value_is_removed(self):
        result = clean_kpis({
            "kpi_scope1_emissions_tco2e_current": "-3",
            "kpi_scope2_emissions_tco2e_current": "10",
        })
        self.assertNotIn("kpi_scope1_emissions_tco2e_current", result)
        self.assertEqual(result["kpi_scope2_emissions_tco2e_current"], 10)

    def test_target_contains_values_and_no_intent_audit(self):
        metadata = {
            "company": "Example Limited",
            "reporting_year": "FY 2024-25",
            "sector": "Industrials",
        }
        kpis = {
            "kpi_scope1_emissions_tco2e_current": 80,
            "kpi_scope1_emissions_tco2e_previous": 100,
            "kpi_scope1_emissions_yoy_reduction_percent": 20,
            "kpi_water_consumption_kl_current": 150,
        }
        target, facts = build_target(metadata, kpis)
        self.assertIn("80 tCO2e", target)
        self.assertIn("reduced by 20%", target)
        self.assertIn("150 KL", target)
        self.assertNotIn("Narrative intent distribution", target)
        self.assertGreaterEqual(len(facts), 4)

    def test_example_has_company_disjoint_split_key(self):
        source = SourceRow(
            row={
                "company": "Example Limited",
                "reporting_year": "fy 2024-25",
                "kpi_scope1_emissions_tco2e_current": "80",
                "kpi_scope1_emissions_tco2e_previous": "100",
            },
            source_file="example.csv",
            source_dataset="BRSR 2024-25 source collection",
            company="Example Limited",
            company_key="EXAMPLE",
            nse_symbol="EXAMPLE",
        )
        example = make_example(source)
        self.assertIsNotNone(example)
        self.assertIn(example["split"], {"train", "validation", "test"})
        self.assertEqual(build_metadata(source)["reporting_year"], "FY 2024-25")


if __name__ == "__main__":
    unittest.main()
