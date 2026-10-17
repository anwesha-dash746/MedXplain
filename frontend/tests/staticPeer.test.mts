// Run with: npm test   (node --experimental-strip-types --test tests/)
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  MAX_QUESTIONS,
  PHYSICIANS,
  RESPONSE_SETS,
  buildThread,
  selectResponseSet,
  type PeerThread,
} from "../lib/staticPeer.ts";

test("five fixed named physicians with the specified specialties and experience", () => {
  assert.equal(new Set(PHYSICIANS.map((p) => p.name)).size, 5);
  assert.deepEqual(
    PHYSICIANS.map((p) => [p.label, p.specialty, p.years]),
    [
      ["Physician 1", "Internal Medicine", 12],
      ["Physician 2", "Emergency Medicine", 10],
      ["Physician 3", "Respiratory Medicine", 9],
      ["Physician 4", "Cardiology", 14],
      ["Physician 5", "General Medicine", 8],
    ],
  );
});

test("each response set has exactly one response per physician", () => {
  for (const set of Object.values(RESPONSE_SETS)) assert.equal(set.responses.length, PHYSICIANS.length);
});

test("diagnosis questions select Set A", () => {
  for (const q of [
    "What is the most likely diagnosis?",
    "What could cause these symptoms?",
    "How should I interpret these findings?",
    "Why is the condition deteriorating?",
  ]) assert.equal(selectResponseSet(q), "diagnosis", q);
});

test("treatment and discharge questions select Set B", () => {
  for (const q of [
    "Is she ready for discharge?",
    "Which medication should be reviewed?",
    "What follow-up and monitoring is needed?",
    "What return precautions belong in the plan?",
    "Is the prescription appropriate for management?",
  ]) assert.equal(selectResponseSet(q), "discharge", q);
});

test("the dominant category wins when both match", () => {
  assert.equal(selectResponseSet("Given the symptoms, what discharge plan, medication and follow-up?"), "discharge");
  assert.equal(selectResponseSet("Do the symptoms and findings change the diagnosis before discharge?"), "diagnosis");
});

test("no match, or a tie, defaults to Set A", () => {
  assert.equal(selectResponseSet("Any thoughts on this case?"), "diagnosis");
  assert.equal(selectResponseSet("Diagnosis or discharge?"), "diagnosis");
});

test("selection is deterministic", () => {
  const q = "What follow-up is needed after discharge?";
  assert.equal(selectResponseSet(q), selectResponseSet(q));
});

test("empty note or question is rejected", () => {
  assert.equal(buildThread("", "What is the diagnosis?", []).ok, false);
  assert.equal(buildThread("Discharge summary text", "   ", []).ok, false);
});

test("exactly two submissions per session, earlier threads kept", () => {
  const threads: PeerThread[] = [];
  for (const q of ["What is the diagnosis?", "What follow-up is needed?"]) {
    const r = buildThread("Discharge summary text", q, threads);
    assert.equal(r.ok, true);
    if (r.ok) threads.push(r.thread);
  }
  assert.equal(threads.length, MAX_QUESTIONS);
  assert.deepEqual(threads.map((t) => t.set), ["diagnosis", "discharge"]);
  const third = buildThread("Discharge summary text", "One more question?", threads);
  assert.equal(third.ok, false);
  assert.equal(threads[0].question, "What is the diagnosis?");
});
