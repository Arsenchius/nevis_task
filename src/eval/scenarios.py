from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.eval.config import CHALLENGE_TAGS, DIFFICULTY_PLAN

# Large diverse name pools; sampled without replacement so every spec gets a
# unique name suggestion and the dataset avoids repeated identities.
_FIRST_NAMES: list[str] = [
    "James", "Olivia", "William", "Emma", "Benjamin", "Sophia", "Lucas",
    "Isabella", "Henry", "Ava", "Alexander", "Charlotte", "Sebastian", "Amelia",
    "Owen", "Abigail", "Theodore", "Grace", "Elijah", "Penelope", "Carter",
    "Riley", "Julian", "Zoey", "Levi", "Nora", "Isaac", "Eleanor",
    "Christopher", "Hannah", "Andrew", "Lillian", "Thomas", "Addison", "Ryan",
    "Ellie", "Nathan", "Stella", "Adrian", "Natalie", "Jonathan", "Zoe",
    "Dominic", "Leah", "Austin", "Hazel", "Justin", "Violet", "Cole",
    "Aurora", "Gavin", "Savannah", "Aaron", "Claire", "Charles", "Skylar",
    "Xavier", "Lucy", "Adam", "Paisley", "Ian", "Everly", "Connor",
    "Caroline", "Caleb", "Genesis", "Robert", "Willow", "Brandon", "Elena",
    "Patrick", "Victoria", "Kevin", "Katherine", "Brian", "Rebecca", "Mark",
    "Amanda", "Steven", "Rachel", "Timothy", "Lauren", "Jeffrey", "Christine",
    "Scott", "Stephanie", "Gregory", "Melissa", "Kenneth", "Jessica", "Raymond",
    "Jennifer", "Dennis", "Lisa", "Walter", "Nancy", "Peter", "Betty",
    "Harold", "Sandra", "Edward", "Dorothy", "George", "Patricia", "Frank",
    "Barbara", "Donald", "Virginia", "Richard", "Margaret", "Arthur", "Martha",
    "Roger", "Gloria", "Gerald", "Joyce", "Stanley", "Shirley", "Howard",
    "Carolyn", "Eugene", "Kathleen", "Carlos", "Maria", "Antonio", "Ana",
    "Miguel", "Carmen", "Jose", "Rosa", "Luis", "Lucia", "Diego", "Camila",
    "Fernando", "Valentina", "Marco", "Priya", "Raj", "Anita", "Vikram",
    "Sunita", "Arun", "Deepa", "Sanjay", "Meera", "Arjun", "Kavya",
    "Wei", "Mei", "Hao", "Yuki", "Hiroshi", "Keiko", "Jin", "Soo",
    "Kwame", "Amara", "Kofi", "Fatima", "Omar", "Zara", "Hassan", "Leila",
    "Tariq", "Nadia", "Samuel", "Grace", "Benjamin", "Faith", "Ezra",
    "Jasmine", "Miles", "Serena", "Felix", "Iris", "Simon", "Vera",
]

_LAST_NAMES: list[str] = [
    "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Martinez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson",
    "Martin", "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez",
    "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen",
    "King", "Wright", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Roberts", "Gomez", "Phillips", "Evans", "Diaz", "Parker", "Cruz",
    "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy",
    "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson",
    "Bailey", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward",
    "Richardson", "Watson", "Brooks", "Chavez", "Wood", "Bennett", "Gray",
    "Mendoza", "Ruiz", "Hughes", "Price", "Alvarez", "Castillo", "Sanders",
    "Patel", "Myers", "Long", "Ross", "Foster", "Jimenez", "Powell",
    "Jenkins", "Perry", "Russell", "Sullivan", "Bell", "Coleman", "Butler",
    "Henderson", "Barnes", "Gonzales", "Fisher", "Vasquez", "Simmons",
    "Romero", "Jordan", "Patterson", "Hamilton", "Graham", "Reynolds",
    "Griffin", "Wallace", "Moreno", "West", "Hayes", "Bryant", "Herrera",
    "Gibson", "Ellis", "Tran", "Medina", "Aguilar", "Hansen", "Ferguson",
    "Hunt", "Shaw", "Rice", "Weaver", "Dixon", "Stone", "Grant", "Warren",
    "Webb", "Spencer", "Burns", "Fox", "Hawkins", "Porter", "Chambers",
    "Soto", "Walsh", "Dunn", "Franco", "Nash", "Snyder", "Watts", "Mills",
    "Ramsey", "Cunningham", "Obrien", "Pearson", "Holt", "Christensen",
    "Walters", "Strickland", "Delgado", "Ingram", "Sherman", "Norris",
]


SCENARIO_ARCHETYPES = [
    {
        "id": "young_high_earner_family",
        "description": "High-earning employed client with partner, young children, childcare costs, pension contributions, and tax-efficiency goals.",
        "likely_sections": [
            "personal_details",
            "household_dependants",
            "employment",
            "income",
            "expenses",
            "pensions_retirement",
            "savings_investments",
            "objectives",
            "risk_profile_preferences",
        ],
        "likely_tags": [
            "client2_present",
            "owner_attribution",
            "numeric_approximate",
            "objectives_free_form",
            "risk_preferences",
        ],
    },
    {
        "id": "pre_retirement_couple",
        "description": "Couple close to retirement discussing Social Security, 401(k)/IRA balances, annuity vs drawdown, no mortgage, and retirement income goals.",
        "likely_sections": [
            "personal_details",
            "household_dependants",
            "employment",
            "income",
            "expenses",
            "pensions_retirement",
            "savings_investments",
            "loans_mortgages",
            "other_assets",
            "objectives",
            "risk_profile_preferences",
            "estate_planning",
        ],
        "likely_tags": [
            "client2_present",
            "joint_assets",
            "numeric_range",
            "correction",
            "negation",
            "objectives_free_form",
            "estate_planning",
        ],
    },
    {
        "id": "self_employed_single",
        "description": "Single self-employed professional with variable income, business expenses, tax concerns, savings, and uncertain retirement planning.",
        "likely_sections": [
            "personal_details",
            "employment",
            "income",
            "expenses",
            "savings_investments",
            "loans_mortgages",
            "other_assets",
            "objectives",
        ],
        "likely_tags": [
            "numeric_range",
            "numeric_approximate",
            "missing_fields",
            "objectives_free_form",
            "owner_attribution",
        ],
    },
    {
        "id": "dual_income_mortgage_household",
        "description": "Two-income household with mortgage, regular expenses, workplace pensions, emergency fund, and medium-term family goals.",
        "likely_sections": [
            "personal_details",
            "household_dependants",
            "employment",
            "income",
            "expenses",
            "pensions_retirement",
            "savings_investments",
            "loans_mortgages",
            "objectives",
        ],
        "likely_tags": [
            "client2_present",
            "joint_assets",
            "owner_attribution",
            "multiple_products",
            "numeric_exact",
        ],
    },
    {
        "id": "inheritance_windfall",
        "description": "Client recently received or expects an inheritance and wants advice on savings, investing, tax efficiency, and estate planning.",
        "likely_sections": [
            "personal_details",
            "employment",
            "income",
            "savings_investments",
            "other_assets",
            "objectives",
            "risk_profile_preferences",
            "estate_planning",
        ],
        "likely_tags": [
            "numeric_approximate",
            "objectives_free_form",
            "risk_preferences",
            "estate_planning",
            "missing_fields",
        ],
    },
    {
        "id": "high_debt_low_savings",
        "description": "Client has credit card or personal loan debt, limited savings, income constraints, and debt-repayment objectives.",
        "likely_sections": [
            "personal_details",
            "employment",
            "income",
            "expenses",
            "savings_investments",
            "loans_mortgages",
            "objectives",
        ],
        "likely_tags": [
            "negation",
            "numeric_shorthand",
            "owner_attribution",
            "missing_fields",
            "objectives_free_form",
        ],
    },
    {
        "id": "retired_widowed_client",
        "description": "Retired client living on pension/Social Security income with savings, estate-planning concerns, and risk aversion.",
        "likely_sections": [
            "personal_details",
            "income",
            "expenses",
            "pensions_retirement",
            "savings_investments",
            "loans_mortgages",
            "objectives",
            "risk_profile_preferences",
            "estate_planning",
        ],
        "likely_tags": [
            "negation",
            "numeric_approximate",
            "estate_planning",
            "risk_preferences",
            "missing_fields",
        ],
    },
    {
        "id": "messy_corrections_privacy",
        "description": "Transcript intentionally includes corrections, adviser mishearings, shorthand amounts, account references mentioned but not extracted, and owner ambiguity.",
        "likely_sections": [
            "personal_details",
            "household_dependants",
            "employment",
            "income",
            "expenses",
            "pensions_retirement",
            "savings_investments",
            "loans_mortgages",
            "objectives",
        ],
        "likely_tags": [
            "correction",
            "numeric_shorthand",
            "numeric_range",
            "privacy_reference",
            "advisor_noise",
            "owner_attribution",
        ],
    },
]


@dataclass(frozen=True)
class DatasetSpec:
    example_id: str
    difficulty: str
    archetype_id: str
    section_targets: list[str]
    challenge_tags: list[str]
    age_band: str
    include_mobile_phone: bool
    include_email: bool
    client1_name: str = ""
    client2_name: str = ""


def build_dataset_specs(seed: int = 42) -> list[DatasetSpec]:
    rng = random.Random(seed)
    specs: list[DatasetSpec] = []
    age_bands = [
        "early_career_1996_2004",
        "young_family_1986_1995",
        "mid_career_1976_1985",
        "pre_retirement_1960_1975",
        "retired_1948_1962",
    ]

    total_specs = sum(DIFFICULTY_PLAN.values())
    first_names = rng.sample(_FIRST_NAMES, k=min(total_specs * 2, len(_FIRST_NAMES)))
    last_names = rng.sample(_LAST_NAMES, k=min(total_specs * 2, len(_LAST_NAMES)))
    name_index = 0

    for difficulty, count in DIFFICULTY_PLAN.items():
        for idx in range(1, count + 1):
            archetype = rng.choice(SCENARIO_ARCHETYPES)
            base_sections = list(archetype["likely_sections"])
            base_tags = list(archetype["likely_tags"])

            if difficulty == "easy":
                tags = sorted(set(base_tags[:2] + ["numeric_exact"]))
                sections = [
                    s
                    for s in base_sections
                    if s not in {"other_assets", "estate_planning"}
                ]
            elif difficulty == "medium":
                tags = sorted(set(base_tags + rng.sample(CHALLENGE_TAGS, k=3)))
                sections = base_sections
            else:
                tags = sorted(
                    set(
                        base_tags
                        + rng.sample(CHALLENGE_TAGS, k=6)
                        + ["correction", "advisor_noise"]
                    )
                )
                sections = sorted(
                    set(
                        base_sections
                        + ["expenses", "objectives", "risk_profile_preferences"]
                    )
                )

            if archetype["id"] == "retired_widowed_client":
                age_band = "retired_1948_1962"
            elif archetype["id"] == "pre_retirement_couple":
                age_band = "pre_retirement_1960_1975"
            elif archetype["id"] in {
                "young_high_earner_family",
                "dual_income_mortgage_household",
            }:
                age_band = rng.choice(["young_family_1986_1995", "mid_career_1976_1985"])
            else:
                age_band = rng.choice(age_bands[:-1])

            c1_first = first_names[name_index % len(first_names)]
            c1_last = last_names[name_index % len(last_names)]
            c2_first = first_names[(name_index + 1) % len(first_names)]
            c2_last = last_names[(name_index + 1) % len(last_names)]
            name_index += 2

            specs.append(
                DatasetSpec(
                    example_id=f"{difficulty}_{idx:03d}_{archetype['id']}",
                    difficulty=difficulty,
                    archetype_id=archetype["id"],
                    section_targets=sections,
                    challenge_tags=tags,
                    age_band=age_band,
                    include_mobile_phone=rng.random() < 0.35,
                    include_email=rng.random() < 0.35,
                    client1_name=f"{c1_first} {c1_last}",
                    client2_name=f"{c2_first} {c2_last}",
                )
            )

    return specs


def summarize_dataset_plan(specs: list[DatasetSpec]) -> dict[str, Any]:
    by_difficulty: dict[str, int] = {}
    by_archetype: dict[str, int] = {}
    by_tag: dict[str, int] = {}
    by_section: dict[str, int] = {}

    for spec in specs:
        by_difficulty[spec.difficulty] = by_difficulty.get(spec.difficulty, 0) + 1
        by_archetype[spec.archetype_id] = by_archetype.get(spec.archetype_id, 0) + 1
        for tag in spec.challenge_tags:
            by_tag[tag] = by_tag.get(tag, 0) + 1
        for section in spec.section_targets:
            by_section[section] = by_section.get(section, 0) + 1

    return {
        "total": len(specs),
        "by_difficulty": by_difficulty,
        "by_archetype": by_archetype,
        "by_tag": dict(sorted(by_tag.items())),
        "by_section": dict(sorted(by_section.items())),
        "contact_targets": {
            "mobile_phone": sum(spec.include_mobile_phone for spec in specs),
            "email": sum(spec.include_email for spec in specs),
        },
        "by_age_band": dict(
            sorted(
                {
                    age_band: sum(spec.age_band == age_band for spec in specs)
                    for age_band in {spec.age_band for spec in specs}
                }.items()
            )
        ),
    }
