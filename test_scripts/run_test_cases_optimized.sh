#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

REGULAR_OPTIMIZED_CASES=(
  "test_cases/GeoGMT/E0/config_ASP_GeoGMT_E0_WC.json"
  "test_cases/GeoGMT/E1/config_ASP_GeoGMT_E1_WC.json"
  "test_cases/GeoGMT/E0/config_ASP_GeoGMT_E0_NC.json"
  "test_cases/GeoGMT/E1/config_ASP_GeoGMT_E1_NC.json"
  "test_cases/GeoGMT/E2/config_ASP_GeoGMT_E2_WC.json"
  "test_cases/GeoGMT/E3/config_ASP_GeoGMT_E3_WC.json"
  "test_cases/GeoGMT/E2/config_ASP_GeoGMT_E2_NC.json"
  "test_cases/GeoGMT/E3/config_ASP_GeoGMT_E3_NC.json"
  "test_cases/GeoGMT/E4/config_ASP_GeoGMT_E4_WC.json"
  "test_cases/GeoGMT/E4/config_ASP_GeoGMT_E4_NC.json"
  "test_cases/GeoEvents/config_ASP_SAT_GeoEvents_NC.json"
  "test_cases/ImageMagick/Example1/config_ASP_ImageMagick_ex1_WC.json"
  "test_cases/ImageMagick/Example1/config_ASP_ImageMagick_Ex1_NC.json"
  "test_cases/ImageMagick/Example2/config_ASP_ImageMagick_Ex2_WC.json"
  "test_cases/ImageMagick/Example2/config_ASP_ImageMagick_Ex2_NC.json"
  "test_cases/metaCascabel/config_ASP_metaCascabel_WC.json"
  "test_cases/metaCascabel/config_ASP_metaCascabel_NC.json"
  "test_cases/Metabolomics/config_ASP_Metabolomics_WC.json"
  "test_cases/Metabolomics/config_ASP_Metabolomics_NC.json"
  "test_cases/defect_concentration/config_ASP_defect_2000_8_WC.json"
  "test_cases/defect_concentration/config_ASP_defect_2000_8_NC.json"
  "test_cases/QuAnGIS/No1/config_ASP_QuAnGIS_no1_WC.json"
  "test_cases/QuAnGIS/No1/config_ASP_QuAnGIS_No1_NC.json"
  "test_cases/QuAnGIS/QGIS/config_ASP_QuAnGIS_QGIS_WC.json"
  "test_cases/QuAnGIS/QGIS/config_ASP_QuAnGIS_QGIS_NC.json"
  "test_cases/QuAnGIS/ArcGIS/config_ASP_QuAnGIS_ArcGIS_NC.json"
  "test_cases/MassSpectrometry/No1/config_ASP_Masspectrometry_No1_extended_WC.json"
  "test_cases/MassSpectrometry/No1/config_ASP_Masspectrometry_No1_original_WC.json"
  "test_cases/MassSpectrometry/No2/config_ASP_Masspectrometry_No2_extended_WC.json"
  "test_cases/MassSpectrometry/No2/config_ASP_Masspectrometry_No2_original_WC.json"
  "test_cases/MassSpectrometry/No3/config_ASP_Masspectrometry_No3_extended_WC.json"
  "test_cases/MassSpectrometry/No3/config_ASP_Masspectrometry_No3_original_WC.json"
  "test_cases/MassSpectrometry/No4/config_ASP_Masspectrometry_No4_extended_WC.json"
  "test_cases/MassSpectrometry/No4/config_ASP_Masspectrometry_No4_original_WC.json"
  "test_cases/biotools/config_ASP_biotools_1000_3_NC.json"
  "test_cases/biotools/config_ASP_biotools_1000_3_WC.json"
  "test_cases/biotools/config_ASP_biotools_5000_3_NC.json"
  "test_cases/biotools/config_ASP_biotools_5000_3_WC.json"
)

run_optimized_matrix() {
  local config="$1"
  echo "==> optimized: $config"
  python -m clindaws "$config" --mode optimized
  echo "==> optimized kcluster: $config"
  python -m clindaws "$config" --mode optimized --decomp kcluster
  echo "==> optimized 1n: $config"
  python -m clindaws "$config" --mode optimized --decomp 1n
}

for config in "${REGULAR_OPTIMIZED_CASES[@]}"; do
  run_optimized_matrix "$config"
done
