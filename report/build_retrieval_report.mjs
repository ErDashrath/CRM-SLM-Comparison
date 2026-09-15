// Rebuild the Phase 4 retrieval workbook from results/retrieval_comparison.json.
// Run with the workspace artifact-tool runtime and set OUTPUT_DIR when needed.
import fs from 'node:fs/promises';
import { SpreadsheetFile, Workbook } from '@oai/artifact-tool';

const project = process.cwd();
const sourcePath = `${project}/results/retrieval_comparison.json`;
const outputDir = process.env.OUTPUT_DIR || `${project}/outputs/phase4-retrieval`;
const source = JSON.parse(await fs.readFile(sourcePath, 'utf8'));
const baseline = source.baseline;
const reranked = source.reranked;
const workbook = Workbook.create();
const summary = workbook.worksheets.add('Summary');
const perQuery = workbook.worksheets.add('Per-query');
const definitions = workbook.worksheets.add('Definitions');
const navy = '#1F4E78';
const lightBlue = '#D9EAF7';
const gray = '#F3F4F6';
const font = 'Arial';

function title(sheet, text, subtitle, endCol) {
  sheet.mergeCells(`A1:${endCol}1`);
  sheet.getRange('A1').values = [[text]];
  sheet.getRange(`A1:${endCol}1`).format = { font: { name: font, size: 16, bold: true, color: navy } };
  sheet.mergeCells(`A2:${endCol}2`);
  sheet.getRange('A2').values = [[subtitle]];
  sheet.getRange(`A2:${endCol}2`).format = { font: { name: font, size: 10, italic: true, color: '#555555' } };
  sheet.showGridLines = false;
}
function header(sheet, range) {
  sheet.getRange(range).format = {
    fill: navy,
    font: { name: font, size: 10, bold: true, color: '#FFFFFF' },
    verticalAlignment: 'center',
    wrapText: true,
    borders: { preset: 'all', style: 'thin', color: '#FFFFFF' },
  };
}
function body(sheet, range) {
  sheet.getRange(range).format = { font: { name: font, size: 10, color: '#222222' }, verticalAlignment: 'center', wrapText: true };
}

title(summary, 'Retrieval evaluation', 'Source: results/retrieval_comparison.json | Six golden CRM queries', 'E');
summary.getRange('A4:B4').values = [['Decision', 'Keep the current lexical retriever for the POC. The deterministic reranker did not improve ranking quality on this golden set.']];
summary.getRange('A4').format = { fill: lightBlue, font: { name: font, bold: true, color: navy } };
summary.getRange('B4').format = { wrapText: true, font: { name: font, size: 10 } };
summary.getRange('A6:E6').values = [['Metric', 'Baseline', 'Deterministic reranker', 'Change', 'Meaning']];
header(summary, 'A6:E6');
const metrics = [
  ['Recall@5', baseline.metrics.recall_at_5, reranked.metrics.recall_at_5, reranked.metrics.recall_at_5 - baseline.metrics.recall_at_5, 'Share of queries with at least one relevant document in the first five results.'],
  ['Recall@10', baseline.metrics.recall_at_10, reranked.metrics.recall_at_10, reranked.metrics.recall_at_10 - baseline.metrics.recall_at_10, 'Share of queries with at least one relevant document in the first ten results.'],
  ['MRR', baseline.metrics.mrr, reranked.metrics.mrr, reranked.metrics.mrr - baseline.metrics.mrr, 'Average reciprocal rank of the first relevant result; higher means relevant evidence appears earlier.'],
  ['nDCG@5', baseline.metrics.ndcg_at_5, reranked.metrics.ndcg_at_5, reranked.metrics.ndcg_at_5 - baseline.metrics.ndcg_at_5, 'Rank-sensitive relevance score for the first five results; higher means better ordering.'],
  ['nDCG@10', baseline.metrics.ndcg_at_10, reranked.metrics.ndcg_at_10, reranked.metrics.ndcg_at_10 - baseline.metrics.ndcg_at_10, 'Rank-sensitive relevance score for the first ten results.'],
];
summary.getRange(`A7:E${6 + metrics.length}`).values = metrics;
body(summary, `A7:E${6 + metrics.length}`);
summary.getRange(`B7:D${6 + metrics.length}`).format.numberFormat = '0.0000';
summary.getRange(`D7:D${6 + metrics.length}`).conditionalFormats.add('colorScale', { colors: ['#F4CCCC', '#FFF2CC', '#D9EAD3'], thresholds: ['min', 0, 'max'] });
summary.getRange('A14:E17').values = [
  ['How to read this', '', '', '', ''],
  ['Recall stayed flat at 1.0000', 'Both systems found relevant evidence for every query within five results.', '', '', ''],
  ['MRR fell from 0.8333 to 0.8056', 'The reranker moved the first relevant result later on average.', '', '', ''],
  ['nDCG@5 fell from 0.8357 to 0.7731', 'The reranker made the top-five ordering worse on this set.', '', '', ''],
];
summary.mergeCells('A14:E14');
summary.mergeCells('B15:E15'); summary.mergeCells('B16:E16'); summary.mergeCells('B17:E17');
summary.getRange('A14:E14').format = { fill: gray, font: { name: font, bold: true, color: navy } };
body(summary, 'A15:E17');
summary.getRange('A15:A17').format.font = { name: font, bold: true, color: navy };
summary.getRange('A:A').format.columnWidth = 22;
summary.getRange('B:B').format.columnWidth = 18;
summary.getRange('C:C').format.columnWidth = 24;
summary.getRange('D:D').format.columnWidth = 14;
summary.getRange('E:E').format.columnWidth = 68;
summary.getRange('A1:E17').format.autofitRows();
summary.freezePanes.freezeRows(6);

const qRows = [['Query ID', 'Question', 'Baseline first relevant rank', 'Reranker first relevant rank', 'Baseline Recall@5', 'Reranker Recall@5', 'Baseline nDCG@5', 'Reranker nDCG@5', 'Rank movement', 'Interpretation']];
for (let i = 0; i < baseline.per_query.length; i++) {
  const b = baseline.per_query[i];
  const r = reranked.per_query[i];
  const bRank = b.first_relevant_rank ?? null;
  const rRank = r.first_relevant_rank ?? null;
  const movement = bRank != null && rRank != null ? rRank - bRank : null;
  const interpretation = movement == null ? 'No relevant result in one or both rankings.' : movement > 0 ? 'Reranker moved the first relevant result later.' : movement < 0 ? 'Reranker moved the first relevant result earlier.' : 'First relevant result stayed at the same rank.';
  qRows.push([b.query_id, b.question, bRank, rRank, b.recall_at['5'], r.recall_at['5'], b.ndcg_at['5'], r.ndcg_at['5'], movement, interpretation]);
}
title(perQuery, 'Per-query retrieval results', 'Ranks and scores from the same golden set used in the summary.', 'J');
perQuery.getRange(`A4:J${3 + qRows.length}`).values = qRows;
header(perQuery, 'A4:J4');
body(perQuery, `A5:J${3 + qRows.length}`);
perQuery.getRange(`E5:I${3 + qRows.length}`).format.numberFormat = '0.0000';
perQuery.getRange(`A5:A${3 + qRows.length}`).format.font = { name: font, bold: true, color: navy };
perQuery.getRange(`I5:I${3 + qRows.length}`).conditionalFormats.add('colorScale', { colors: ['#D9EAD3', '#FFF2CC', '#F4CCCC'], thresholds: ['min', 0, 'max'] });
perQuery.getRange('A:A').format.columnWidth = 14;
perQuery.getRange('B:B').format.columnWidth = 48;
perQuery.getRange('C:D').format.columnWidth = 19;
perQuery.getRange('E:H').format.columnWidth = 15;
perQuery.getRange('I:I').format.columnWidth = 14;
perQuery.getRange('J:J').format.columnWidth = 44;
perQuery.getRange(`A1:J${3 + qRows.length}`).format.autofitRows();
perQuery.freezePanes.freezeRows(4);

title(definitions, 'Metric definitions and scope', 'Definitions are included so the numerical result is interpretable without knowing retrieval terminology.', 'D');
definitions.getRange('A4:D4').values = [['Metric / field', 'Definition', 'Interpretation', 'Source']];
header(definitions, 'A4:D4');
definitions.getRange('A5:D11').values = [
  ['Recall@5', 'Whether at least one graded relevant document appears in the top five results.', '1.0000 means every golden query found relevant evidence within five results.', 'data/retrieval_eval.jsonl'],
  ['MRR', 'Mean reciprocal rank of the first relevant document across queries.', 'Higher is better because relevant evidence appears earlier.', 'results/retrieval_comparison.json'],
  ['nDCG@5', 'Normalized discounted cumulative gain over the top five ranks.', 'Higher is better because highly relevant documents are rewarded for appearing earlier.', 'results/retrieval_comparison.json'],
  ['Rank movement', 'Reranker first-relevant rank minus baseline first-relevant rank.', 'Positive values mean the reranker moved relevant evidence later.', 'results/retrieval_comparison.json'],
  ['Golden set', 'Six manually labeled CRM questions with relevant document IDs.', 'This is a small POC benchmark; results should be treated as directional evidence.', 'data/retrieval_eval.jsonl'],
  ['Baseline', 'Current hybrid lexical retriever used by the chat path.', 'The selected POC default after this comparison.', 'common/crm_retrieval.py'],
  ['Deterministic reranker', 'Transparent weighted score layered on top of baseline candidates.', 'Experimental candidate; not selected because its metrics declined.', 'common/retrieval_reranker.py'],
];
body(definitions, 'A5:D11');
definitions.getRange('A5:A11').format.font = { name: font, bold: true, color: navy };
definitions.getRange('A:A').format.columnWidth = 24;
definitions.getRange('B:B').format.columnWidth = 65;
definitions.getRange('C:C').format.columnWidth = 58;
definitions.getRange('D:D').format.columnWidth = 34;
definitions.getRange('A1:D11').format.autofitRows();
definitions.freezePanes.freezeRows(4);

await fs.mkdir(outputDir, { recursive: true });
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(`${outputDir}/retrieval-evaluation.xlsx`);
console.log(`Wrote ${outputDir}/retrieval-evaluation.xlsx`);
