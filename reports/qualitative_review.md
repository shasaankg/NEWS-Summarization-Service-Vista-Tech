# Qualitative review of generated summaries

The five examples below are the first five articles in the seed-42 evaluation
sample. They were selected by sample order, not by ROUGE score. This is a
manual, descriptive inspection of the source article, human reference, and
unchanged model output. It is not a statistically representative factuality study.
Full text and generated output are in `sample_summaries.md` and
`sample_summaries.json`.

| Example / article ID | What the summary captures | Observed limitation |
|---|---|---|
| 1 / `a10fd75e110ddf8b5d2a283b9ba65e204808d669` | Henderson's tweet to Andrews, the earlier exchange, and Stoll's arrest. | Omits Henderson's earlier suspension, which explains the retaliation. Introduces Stoll without clearly explaining his relationship to Andrews. The quoted material makes the summary less self-contained. |
| 2 / `046b7d1b181fcb7e80b95f58eda938cec7a0735c` | The allegation against Ian Walters, the crash, Tracy Walters's death, and the prosecution's account. | Ends in an unfinished clause at the output length limit. Describes the vehicle as her car although the article identifies it as her husband's. Omits his denial and defence account. This output should not be treated as a balanced account of the trial. |
| 3 / `9f3b0634d37e6b7d5de9195313af5a832e2b2e9d` | Habana's decisive try and Halfpenny's points. | Incorrectly says Leinster were reduced to 14 men; the source says Toulon. This is a clear team-attribution error. The input was truncated, but the correct team is stated near the beginning, so truncation alone does not explain the error. It also repeats a source statement about the game's only try even though a later passage describes O'Brien scoring. |
| 4 / `e01ab56f3f5e8da0ad856fe014317ee2eddd8e50` | Scholfield's planned Grand National ride, the date, and earlier racing results. | Repeats the year 2013 and shifts to the horse's age without clearly naming the horse again. Omits the planned visit to Ireland included in the reference. |
| 5 / `7f12cf929a31457ec4abd62f2e6f04436b893a04` | The six Leeds players withdrawing, Redfearn's reaction, and Cherry's call for dismissal. | Lists four of the six players and omits broader club context. The central claims in this example are supported by the source. |

## Implications

The model often preserves the main news event and named entities, but fluent
output does not guarantee accurate attribution. A summary can have substantial
word overlap while assigning an event to the wrong team or losing important
qualifications. ROUGE therefore complements rather than replaces source review.

The fixed generation budget also permits unfinished final sentences. No manual
correction or sentence trimming has been applied to the saved predictions; the
reported scores and examples describe the actual configured model outputs.
No generation settings were tuned after inspecting these test examples.

Readers should verify generated claims against the article. The scope remains
pretrained inference and evaluation; additional training, factuality models,
and other extensions are not part of this project.
