"""Pre-built sample corpora so the demo is interesting before any upload.

Every corpus is written for the grader demo, not for retrieval: each one mixes
answer-bearing passages with passages that share vocabulary with the query but
do not answer it. A pure embedding retriever will happily return those decoys —
that is the failure the ON/OFF toggle exists to show.

Dependency-free on purpose (stdlib only) so this file can be copied into a
Hugging Face Space without dragging the repo along.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SampleCorpus:
    """A tiny hand-written corpus plus the query the demo is built around."""

    key: str
    title: str
    description: str
    passages: list[str] = field(default_factory=list)
    query: str = ""
    expected_gate: str = ""
    """What the gate is *designed* to do on this query. A design expectation
    about decoy rejection, not a measured score for the model."""


COLAB_T4_NOTES = SampleCorpus(
    key="colab-t4",
    title="Colab T4 fine-tuning notes",
    description=(
        "Eight notes about fitting a fine-tune into a free-tier GPU. Four answer "
        "the query; four talk about GPUs and plans that are not the T4."
    ),
    passages=[
        "Free Colab sessions are assigned an NVIDIA T4 with 16 GB of VRAM. That "
        "number, not the parameter count of your model, is what decides the "
        "largest per-device batch size you can use, and it is why tutorials that "
        "fit on a paid instance OOM on the free tier.",
        "QLoRA keeps the base weights frozen and loads them in 4-bit NF4, so a "
        "1.5B-parameter model occupies roughly a third of its fp16 footprint. "
        "Only the LoRA adapters are trained, and those stay in full precision, "
        "which is where most of the memory savings come from.",
        "Gradient checkpointing trades compute for memory: activations are thrown "
        "away during the forward pass and recomputed during the backward pass. "
        "It slows training down by roughly a third but is the single biggest "
        "lever when both batch size and sequence length are already large.",
        "The A100 available on paid Colab carries 80 GB of HBM, so scripts that "
        "OOM on a T4 will run unquantized with large batches on that card. The "
        "hardware is a different generation entirely, so timings do not carry "
        "over between the two.",
        "The NVIDIA L4 is the newer Ada Lovelace card that appears on some "
        "Colab Pro tiers. It offers 24 GB of memory and much higher bandwidth "
        "than the T4, which shifts the memory budget in a way that has to be "
        "re-measured rather than assumed from T4 numbers.",
        "A Colab Pro subscription buys more GPU-hours and shorter idle timeouts. "
        "It does not change the hardware that free sessions receive, so the "
        "memory ceiling a free session runs under stays exactly the same.",
        "The T4 is a Turing-generation card with 16 GB of GDDR6 memory. It has "
        "no tensor cores, so bf16 matrix multiplies are emulated and can run "
        "slower than fp16 rather than faster on this hardware.",
        "Weights & Biases logging adds negligible overhead to a training loop, "
        "but the sweeps feature needs a paid plan once you want more than a "
        "handful of concurrent runs tracked against the same project.",
    ],
    query=(
        "How much VRAM does a free Colab T4 have, and what do I trade away to fit "
        "a fine-tune into it?"
    ),
    expected_gate=(
        "Keep the 16 GB T4, QLoRA 4-bit and gradient-checkpointing notes; the "
        "A100, L4, Colab Pro and W&B notes are decoys that match on vocabulary."
    ),
)

ESPRESSO_TROUBLESHOOTING = SampleCorpus(
    key="espresso",
    title="Home espresso troubleshooting",
    description=(
        "Eight notes from a home barista's log. Five bear on a thin, fast shot; "
        "three are maintenance tasks that share words with the query."
    ),
    passages=[
        "Grind size sets flow rate. A coarse grind leaves large gaps for water to "
        "rush through, so the shot pours in seconds, the crema looks pale, and "
        "the cup tastes thin and sour at the same time. Finer by one notch is "
        "usually the whole fix.",
        "Brew pressure should reach about nine bars at the group head. If the "
        "pump never gets there the puck is too coarse to resist the water, and if "
        "the gauge sits pinned at zero the puck may be too tight or the basket "
        "blocked.",
        "Water temperature belongs between 92 and 94 degrees. Cooler water "
        "under-extracts and gives a sharp, salty cup; hotter water pushes "
        "bitterness out of the grounds and speeds up the pour noticeably.",
        "Channeling means part of the puck erodes early and water escapes down "
        "one side instead of through the bed. The signs are a fast pour with one "
        "channel in the stream and a spent puck that comes out with a hole in it.",
        "Steaming milk is a separate skill from pulling espresso: introduce air "
        "during the first third of the pitcher, then drop the wand below the "
        "surface to roll the milk until it reaches about 60 degrees. Pushing the "
        "steam any harder just makes large fast bubbles you cannot texture.",
        "Descale the machine every one to three months depending on how hard "
        "your water is. Symptoms of scale are a slow flow rate and a rising "
        "temperature reading, not changes in how fast the shot itself comes out.",
        "A burr grinder matters more than most upgrades because blade grinders "
        "produce a wide range of particle sizes. Fines from blades clog the puck "
        "and stall the shot; a consistent grind keeps the pressure steady.",
        "A leak from the group head is a gasket problem, not a taste problem. "
        "Replace the gasket when water drips from under the portafilter after "
        "the shot is pulled, and check that the basket is seated before blaming "
        "the grinder.",
    ],
    query="My espresso comes out thin and gushes out far too fast. What should I change?",
    expected_gate=(
        "Keep grind, pressure, temperature and channeling; descaling, gasket "
        "and milk-steaming notes are maintenance tasks, not answers."
    ),
)

AMSTERDAM_TRAVEL = SampleCorpus(
    key="amsterdam",
    title="Amsterdam travel desk",
    description=(
        "Eight notes from a travel desk. Four are about trains and paying; four "
        "are about opening hours and sights, which retrieve well on the word "
        "'hours'."
    ),
    passages=[
        "Night trains from Utrecht, Rotterdam and Den Haag reach Amsterdam "
        "Centraal between roughly 00:30 and 02:30. The last direct service is "
        "around 02:00, so a flight the next morning usually needs a hotel near "
        "the station rather than a night bus.",
        "You can tap a contactless bank card on the platform gates and buy a "
        "one-hour ticket without an account. Each tap starts a new hour, and "
        "checking in and out with the same card keeps the fare calculated per "
        "journey.",
        "The OV-chipkaart is being replaced by OVpay: a debit card or phone "
        "wallet works in the meantime, and OV-fiets bikes unlock with the same "
        "tap. Paper tickets are still sold at the machine inside the station.",
        "The Rijksmuseum opens daily at 09:00 and closes at 17:00, with last "
        "entry an hour before closing. The Anne Frank House sells timed-entry "
        "tickets online only and is often booked out weeks ahead.",
        "Canal cruises leave from behind the station every thirty minutes in "
        "season. The evening departures run until about 22:00 and are the "
        "quieter way to see the lit bridges than walking the canal ring.",
        "OV-fiets rental is included in most rail passes, otherwise it costs a "
        "few euros per day. Availability drops sharply during the evening "
        "commute when everyone is cycling home at the same time.",
        "Amsterdam Centraal is about fifteen minutes by direct train from "
        "Schiphol airport, and every twelve minutes or so outside the peaks. "
        "Night buses also run from the airport but take close to an hour.",
        "Trams 26 and 27 leave Centraal for the museums quarter, while tram 5 "
        "runs along the canal ring past the Anne Frank House. Trams stop for "
        "traffic, so the metro is faster when a schedule matters.",
    ],
    query="How late do trains run in Amsterdam, and can I pay with a bank card?",
    expected_gate=(
        "Keep the night-train, tap-to-pay and OVpay notes; the museum hours and "
        "cruise notes are decoys that match on 'hours' and 'run'."
    ),
)

CORPORA: tuple[SampleCorpus, ...] = (
    COLAB_T4_NOTES,
    ESPRESSO_TROUBLESHOOTING,
    AMSTERDAM_TRAVEL,
)


def get_corpus(key: str) -> SampleCorpus:
    """Look up a corpus by key or by display title.

    Both are accepted because Streamlit round-trips a `format_func` result back
    through the same function when restoring widget state.
    """
    for corpus in CORPORA:
        if key in (corpus.key, corpus.title):
            return corpus
    raise KeyError(f"Unknown sample corpus {key!r}; known: {[c.key for c in CORPORA]}")


def corpus_title(value: str) -> str:
    """Idempotent display name for a corpus key or title (Streamlit format_func)."""
    return get_corpus(value).title
