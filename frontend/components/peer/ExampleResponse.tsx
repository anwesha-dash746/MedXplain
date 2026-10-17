"use client";

/**
 * A fixed, hypothetical example of what a colleague's response could look like,
 * shown to the requesting clinician. It is the same text for every case of a
 * given consultation type: it does not read the case, is not written by a
 * clinician, and gives no patient-specific advice (no diagnoses, drugs or
 * doses). It exists to show the shape of a useful peer opinion while no real
 * colleague has replied.
 */

interface Example {
  interpretation: string;
  considerations: string[];
  questions: string[];
  limitations: string;
  references: string;
}

const GENERAL_LIMITATIONS =
  "A real colleague would be working only from the de-identified summary, without examining the patient or seeing the full record.";

const EXAMPLES: Record<string, Example> = {
  second_opinion: {
    interpretation:
      "From the summary alone, the working diagnosis seems broadly consistent with the findings described, though the summary by itself cannot confirm it.",
    considerations: [
      "Check that the timeline of symptoms, findings and results supports the working diagnosis.",
      "Ask whether any reasonable alternative explanation has not yet been excluded.",
      "Review whether any investigations are still pending and who will follow them up.",
    ],
    questions: ["What has changed since admission?", "Is anything missing from the summary that would change the picture?"],
    limitations: GENERAL_LIMITATIONS,
    references: "Local and national guidelines for the condition in question.",
  },
  complex_case: {
    interpretation:
      "Several active problems appear to interact here, so it may help to rank them by immediate risk and agree which one drives today's decisions.",
    considerations: [
      "List the active problems and the current plan for each.",
      "Look for treatments that could help one problem while worsening another.",
      "Consider whether a multidisciplinary discussion would be useful.",
    ],
    questions: ["Which problem worries you most right now?", "What are the patient's own priorities and goals of care?"],
    limitations: GENERAL_LIMITATIONS,
    references: "Relevant specialty guidelines for each active problem.",
  },
  diagnostic: {
    interpretation:
      "The findings described fit more than one explanation, so a structured differential diagnosis may be more useful than committing to one now.",
    considerations: [
      "Group the findings and note which explanations account for most of them.",
      "Identify the dangerous diagnoses that should be ruled out first.",
      "Choose the investigation most likely to change management.",
    ],
    questions: ["Which result would most change your thinking?", "How has the patient responded to treatment so far?"],
    limitations: GENERAL_LIMITATIONS,
    references: "Diagnostic pathways in local or national guidance for the presenting problem.",
  },
  treatment_review: {
    interpretation:
      "The overall direction of the plan sounds reasonable from the summary. The main value of a review is checking the details against the patient's current state.",
    considerations: [
      "Re-check that each current treatment still has a clear indication.",
      "Review interactions, kidney and liver function, and any monitoring each treatment needs.",
      "Agree in advance what would prompt a change of plan.",
    ],
    questions: ["What goal is each treatment meant to achieve?", "Have the patient's preferences been discussed?"],
    limitations: GENERAL_LIMITATIONS,
    references: "Current treatment guidelines and the local formulary.",
  },
  follow_up: {
    interpretation:
      "Arrangements after discharge seem to be the main concern, and making responsibilities explicit usually matters most.",
    considerations: [
      "Name who will follow the patient up, and by when.",
      "List pending results and who is responsible for acting on them.",
      "Make sure the patient knows the warning signs and whom to contact.",
    ],
    questions: ["Who is the patient's main point of contact after discharge?", "Are there barriers to attending follow-up?"],
    limitations: GENERAL_LIMITATIONS,
    references: "Local discharge and transfer-of-care policies.",
  },
};

export function ExampleResponse({ consultType }: { consultType: string }) {
  const ex = EXAMPLES[consultType] ?? EXAMPLES.second_opinion;
  return (
    <section className="rounded-lg border-2 border-dashed border-amber-300 bg-amber-50/60 p-4" aria-label="Hypothetical example response">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-semibold text-amber-950">Hypothetical example response</h3>
        <span className="chip bg-amber-200 text-amber-950">Static example · not written by a clinician</span>
      </div>
      <p className="mt-1 text-xs text-amber-900">
        This is fixed sample text for this type of consultation, so you can see what a colleague&rsquo;s
        reply might cover. It has not read your case and is not advice about your patient.
      </p>
      <dl className="mt-3 space-y-2 text-sm text-slate-800">
        <div>
          <dt className="text-xs font-semibold text-slate-600">Interpretation</dt>
          <dd>{ex.interpretation}</dd>
        </div>
        <div>
          <dt className="text-xs font-semibold text-slate-600">Relevant considerations</dt>
          <dd>
            <ul className="list-disc pl-5">
              {ex.considerations.map((c) => (
                <li key={c}>{c}</li>
              ))}
            </ul>
          </dd>
        </div>
        <div>
          <dt className="text-xs font-semibold text-slate-600">Questions back to you</dt>
          <dd>
            <ul className="list-disc pl-5">
              {ex.questions.map((q) => (
                <li key={q}>{q}</li>
              ))}
            </ul>
          </dd>
        </div>
        <div>
          <dt className="text-xs font-semibold text-slate-600">Limitations</dt>
          <dd>{ex.limitations}</dd>
        </div>
        <div>
          <dt className="text-xs font-semibold text-slate-600">References</dt>
          <dd>{ex.references}</dd>
        </div>
      </dl>
    </section>
  );
}
