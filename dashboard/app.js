(function () {
  "use strict";

  const data = window.BENCHMARK_DASHBOARD_DATA;
  const rows = data.rows;
  const kpis = data.kpis;
  const disclosures = data.disclosures;

  const $ = (id) => document.getElementById(id);

  const companySelect = $("companySelect");
  const yearSelect = $("yearSelect");
  const benchmarkMode = $("benchmarkMode");
  const sectorSelect = $("sectorSelect");
  const peerSearch = $("peerSearch");
  const peerSelect = $("peerSelect");
  const sectorField = $("sectorField");
  const peerField = $("peerField");

  function clean(value) {
    if (value === null || value === undefined) return "";
    return String(value).replace(/\s+/g, " ").trim();
  }

  function normalizeCompany(value) {
    return clean(value)
      .toUpperCase()
      .replace(/\b(LIMITED|LTD|PRIVATE|PVT|INDIA|CO|COMPANY)\b/g, " ")
      .replace(/[^A-Z0-9]+/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function numberValue(value) {
    if (value === null || value === undefined || value === "") return null;
    const match = String(value).replace(/,/g, "").match(/-?\d+(\.\d+)?/);
    if (!match) return null;
    const num = Number(match[0]);
    return Number.isFinite(num) ? num : null;
  }

  function truthy(value) {
    const text = clean(value).toLowerCase();
    return !["", "false", "0", "no", "nan", "null", "none"].includes(text);
  }

  function integerValue(value) {
    const num = numberValue(value);
    return num === null ? 0 : Math.round(num);
  }

  function formatNumber(value, unit) {
    if (value === null || value === undefined || Number.isNaN(value)) return "Not disclosed";
    const abs = Math.abs(value);
    const formatted = abs >= 1000
      ? value.toLocaleString(undefined, { maximumFractionDigits: 2 })
      : value.toLocaleString(undefined, { maximumFractionDigits: 2 });
    return `${formatted}${unit ? ` ${unit}` : ""}`;
  }

  function uniqueSorted(items) {
    return [...new Set(items.filter(Boolean))].sort((a, b) => a.localeCompare(b));
  }

  function option(value, label) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = label || value;
    return opt;
  }

  function populateInitialControls() {
    const companyMap = new Map();
    rows.forEach((row) => {
      const label = row.company_display || row.company || "Unknown Company";
      const key = `${label}|||${row.company || label}`;
      if (!companyMap.has(key)) companyMap.set(key, { label, value: row.company || label });
    });
    [...companyMap.values()]
      .sort((a, b) => a.label.localeCompare(b.label))
      .forEach((item) => companySelect.appendChild(option(item.value, item.label)));

    uniqueSorted(rows.map((row) => row.reporting_year)).reverse()
      .forEach((year) => yearSelect.appendChild(option(year)));

    uniqueSorted(rows.map((row) => row.sector)).forEach((sector) => {
      sectorSelect.appendChild(option(sector));
    });

    yearSelect.value = rows.find((row) => row.reporting_year === "FY 2024-25") ? "FY 2024-25" : yearSelect.value;
    const defaultRow = rows.find((row) => row.reporting_year === yearSelect.value && integerValue(row.brsr_report_count) > 0)
      || rows.find((row) => integerValue(row.report_count) > 0)
      || rows.find((row) => clean(row.company).includes("TCS"));
    companySelect.value = defaultRow?.company || companySelect.value;
    updatePeerList();
  }

  function targetRow() {
    const selectedCompany = companySelect.value;
    const selectedYear = yearSelect.value;
    const key = normalizeCompany(selectedCompany);
    const candidates = rows.filter((row) => {
      return row.reporting_year === selectedYear
        && (normalizeCompany(row.company).includes(key) || normalizeCompany(row.company_display).includes(key));
    });
    return candidates.sort((a, b) => (b.benchmark_quality_score || 0) - (a.benchmark_quality_score || 0))[0];
  }

  function updatePeerList() {
    const year = yearSelect.value;
    const target = targetRow();
    const targetNorm = target ? normalizeCompany(target.company) : "";
    peerSelect.innerHTML = "";
    const filter = normalizeCompany(peerSearch.value);
    rows
      .filter((row) => row.reporting_year === year)
      .filter((row) => normalizeCompany(row.company) !== targetNorm)
      .filter((row) => !filter || normalizeCompany(`${row.company_display} ${row.company}`).includes(filter))
      .sort((a, b) => clean(a.company_display).localeCompare(clean(b.company_display)))
      .slice(0, 350)
      .forEach((row) => {
        const opt = option(row.company, `${row.company_display || row.company} (${row.sector || "Unknown"})`);
        peerSelect.appendChild(opt);
      });
  }

  function currentPeers(target) {
    const year = yearSelect.value;
    const targetNorm = normalizeCompany(target.company);
    if (benchmarkMode.value === "custom") {
      const selected = [...peerSelect.selectedOptions].map((opt) => opt.value);
      const selectedKeys = selected.map(normalizeCompany);
      return rows
        .filter((row) => row.reporting_year === year)
        .filter((row) => selectedKeys.some((key) => normalizeCompany(row.company).includes(key)))
        .filter((row) => normalizeCompany(row.company) !== targetNorm);
    }
    const sector = sectorSelect.value || target.sector;
    return rows
      .filter((row) => row.reporting_year === year && row.sector === sector)
      .filter((row) => normalizeCompany(row.company) !== targetNorm);
  }

  function compareKpi(targetValue, peerValues, lowerIsBetter) {
    const sorted = [...peerValues].sort((a, b) => a - b);
    const sum = peerValues.reduce((acc, value) => acc + value, 0);
    const median = sorted.length % 2
      ? sorted[Math.floor(sorted.length / 2)]
      : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
    const average = sum / peerValues.length;
    const min = sorted[0];
    const max = sorted[sorted.length - 1];
    const rank = lowerIsBetter
      ? 1 + peerValues.filter((value) => value < targetValue).length
      : 1 + peerValues.filter((value) => value > targetValue).length;
    const interpretation = lowerIsBetter
      ? (targetValue <= median ? "better than or equal to peer median" : "worse than peer median")
      : (targetValue >= median ? "better than or equal to peer median" : "worse than peer median");
    return { average, median, min, max, rank, interpretation };
  }

  function buildReport(target, peers) {
    const kpiComparisons = [];
    const missingKpis = [];
    const disclosureGaps = [];

    Object.entries(kpis).forEach(([column, meta]) => {
      const targetValue = numberValue(target[column]);
      const peerValues = peers.map((row) => numberValue(row[column])).filter((value) => value !== null);
      if (targetValue === null) {
        if (peerValues.length) {
          missingKpis.push({
            label: meta.label,
            adoption: 100 * peerValues.length / Math.max(peers.length, 1),
            count: peerValues.length,
            suggestion: `Track and disclose ${meta.label.toLowerCase()} to match peer reporting practice.`
          });
        }
        return;
      }
      if (!peerValues.length) return;
      const stats = compareKpi(targetValue, peerValues, meta.lower_is_better);
      kpiComparisons.push({ column, meta, targetValue, peerCount: peerValues.length, ...stats });
    });

    Object.entries(disclosures).forEach(([column, label]) => {
      if (truthy(target[column]) || !peers.length) return;
      const disclosing = peers.filter((row) => truthy(row[column]));
      const adoption = 100 * disclosing.length / peers.length;
      if (adoption >= 60) {
        disclosureGaps.push({
          label,
          adoption,
          examples: uniqueSorted(disclosing.map((row) => row.company_display || row.company)).slice(0, 5),
          suggestion: `Add a clearer ${label.toLowerCase()} disclosure because it is common in the selected peer group.`
        });
      }
    });

    return { target, peers, kpiComparisons, missingKpis, disclosureGaps };
  }

  function renderReport(report) {
    const peerLabel = benchmarkMode.value === "custom" ? "custom peer group" : `${sectorSelect.value || report.target.sector} sector`;
    $("summaryTitle").textContent = `${report.target.company_display || report.target.company} | ${report.target.reporting_year}`;
    $("summarySubtitle").textContent = `Compared against ${peerLabel}. Source rows: ${data.row_count.toLocaleString()}.`;
    $("reportCount").textContent = integerValue(report.target.report_count);
    $("peerCount").textContent = report.peers.length;
    $("kpiCount").textContent = report.kpiComparisons.length;
    $("gapCount").textContent = report.missingKpis.length + report.disclosureGaps.length;

    renderKpiTable(report);
    renderMissing(report);
    renderDisclosures(report);
    renderReports(report);
    renderChart(report);
  }

  function renderKpiTable(report) {
    const tbody = $("kpiTable");
    tbody.innerHTML = "";
    if (!report.kpiComparisons.length) {
      tbody.innerHTML = `<tr><td colspan="7">No comparable KPI values found for this selection.</td></tr>`;
      return;
    }
    report.kpiComparisons.forEach((item) => {
      const unit = item.meta.unit || "";
      const tr = document.createElement("tr");
      const statusClass = item.interpretation.startsWith("better") ? "status-good" : "status-bad";
      tr.innerHTML = `
        <td>${item.meta.label}</td>
        <td>${formatNumber(item.targetValue, unit)}</td>
        <td>${formatNumber(item.median, unit)}</td>
        <td>${formatNumber(item.average, unit)}</td>
        <td>${formatNumber(item.min, unit)} to ${formatNumber(item.max, unit)}</td>
        <td>${item.rank}/${item.peerCount + 1}</td>
        <td class="${statusClass}">${item.interpretation}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  function renderMissing(report) {
    const list = $("missingList");
    list.innerHTML = "";
    if (!report.missingKpis.length) {
      list.innerHTML = `<div class="empty">No missing KPI opportunities detected for this peer selection.</div>`;
      return;
    }
    report.missingKpis.forEach((item) => {
      const div = document.createElement("div");
      div.className = "item";
      div.innerHTML = `
        <strong>${item.label}</strong>
        <p>${item.adoption.toFixed(1)}% of selected peers disclose this KPI (${item.count} companies).</p>
        <p>${item.suggestion}</p>
      `;
      list.appendChild(div);
    });
  }

  function renderDisclosures(report) {
    const list = $("disclosureList");
    list.innerHTML = "";
    if (!report.disclosureGaps.length) {
      list.innerHTML = `<div class="empty">No high-adoption disclosure gaps detected for this peer selection.</div>`;
      return;
    }
    report.disclosureGaps.forEach((item) => {
      const div = document.createElement("div");
      div.className = "item";
      div.innerHTML = `
        <strong>${item.label}</strong>
        <p>${item.adoption.toFixed(1)}% peer adoption.</p>
        <p>Examples: ${item.examples.join(", ") || "Not available"}</p>
        <p>${item.suggestion}</p>
      `;
      list.appendChild(div);
    });
  }

  function renderReports(report) {
    const list = $("reportList");
    list.innerHTML = "";
    const target = report.target;
    const peers = report.peers;
    const peerReportCount = peers.filter((row) => integerValue(row.report_count) > 0).length;
    const peerBrsrCount = peers.filter((row) => integerValue(row.brsr_report_count) > 0).length;
    const peerAnnualCount = peers.filter((row) => integerValue(row.annual_extracted_source_count) > 0 || integerValue(row.annual_report_count) > 0).length;
    const totalPeers = Math.max(peers.length, 1);

    const targetDiv = document.createElement("div");
    targetDiv.className = "item";
    targetDiv.innerHTML = `
      <strong>Target company evidence</strong>
      <p>BRSR reports connected: ${integerValue(target.brsr_report_count)}</p>
      <p>Annual report sources connected: ${integerValue(target.annual_report_count) + integerValue(target.annual_extracted_source_count)}</p>
      <p>Total connected sources: ${integerValue(target.report_count)}</p>
      <p>Types: ${clean(target.report_types_available) || "Not connected yet"}</p>
      <p>Example source: ${clean(target.example_report_path) || "Not available"}</p>
    `;
    list.appendChild(targetDiv);

    const peerDiv = document.createElement("div");
    peerDiv.className = "item";
    peerDiv.innerHTML = `
      <strong>Selected peer evidence coverage</strong>
      <p>${peerReportCount}/${peers.length} peers have at least one connected report/evidence source.</p>
      <p>${peerBrsrCount}/${peers.length} peers have connected BRSR PDFs.</p>
      <p>${peerAnnualCount}/${peers.length} peers have annual report or annual extracted evidence.</p>
      <p>Coverage rate: ${(100 * peerReportCount / totalPeers).toFixed(1)}%</p>
    `;
    list.appendChild(peerDiv);
  }

  function renderChart(report) {
    const chart = $("chart");
    chart.innerHTML = "";
    const items = report.kpiComparisons.slice(0, 7);
    if (!items.length) {
      chart.innerHTML = `<div class="empty">No chartable KPI comparisons for this selection.</div>`;
      return;
    }
    items.forEach((item) => {
      const max = Math.max(Math.abs(item.targetValue), Math.abs(item.median), 1);
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `
        <div class="bar-label">${item.meta.label}</div>
        <div class="bar-track">
          <div class="bar"><span>Company</span><div class="bar-fill" style="width:${Math.max(2, Math.abs(item.targetValue) / max * 100)}%"></div></div>
          <div class="bar"><span>Median</span><div class="bar-fill peer" style="width:${Math.max(2, Math.abs(item.median) / max * 100)}%"></div></div>
        </div>
      `;
      chart.appendChild(row);
    });
  }

  function runBenchmark() {
    const target = targetRow();
    if (!target) return;
    const peers = currentPeers(target);
    const report = buildReport(target, peers);
    renderReport(report);
  }

  function setupEvents() {
    companySelect.addEventListener("change", () => {
      const target = targetRow();
      if (target && target.sector) sectorSelect.value = target.sector;
      updatePeerList();
      runBenchmark();
    });
    yearSelect.addEventListener("change", () => {
      updatePeerList();
      runBenchmark();
    });
    sectorSelect.addEventListener("change", runBenchmark);
    peerSearch.addEventListener("input", updatePeerList);
    peerSelect.addEventListener("change", runBenchmark);
    benchmarkMode.addEventListener("change", () => {
      const custom = benchmarkMode.value === "custom";
      sectorField.classList.toggle("hidden", custom);
      peerField.classList.toggle("hidden", !custom);
      updatePeerList();
      runBenchmark();
    });
    $("runButton").addEventListener("click", runBenchmark);
    document.querySelectorAll(".tab").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
        button.classList.add("active");
        $(`tab-${button.dataset.tab}`).classList.add("active");
      });
    });
  }

  populateInitialControls();
  const target = targetRow();
  if (target && target.sector) sectorSelect.value = target.sector;
  setupEvents();
  runBenchmark();
})();
