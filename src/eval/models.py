from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Owner(str, Enum):
    CLIENT1 = "client1"
    CLIENT2 = "client2"
    JOINT = "joint"
    HOUSEHOLD = "household"
    UNKNOWN = "unknown"


class Frequency(str, Enum):
    ANNUAL = "annual"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    WEEKLY = "weekly"
    DAILY = "daily"
    ONE_OFF = "one_off"
    UNKNOWN = "unknown"


class NetGross(str, Enum):
    NET = "net"
    GROSS = "gross"
    UNKNOWN = "unknown"


class EmploymentStatus(str, Enum):
    EMPLOYED = "employed"
    SELF_EMPLOYED = "self_employed"
    RETIRED = "retired"
    UNEMPLOYED = "unemployed"
    PART_TIME = "part_time"
    FULL_TIME = "full_time"
    UNKNOWN = "unknown"


# CIF section models


class MoneyAmount(BaseModel):
    normalized_amount: Optional[float] = Field(
        default=None,
        description="Best single normalized amount in the stated currency. For ranges, use midpoint unless a better representative value is implied.",
    )
    lower_bound: Optional[float] = Field(
        default=None,
        description="Lower bound for a stated range or approximate amount, if inferable.",
    )
    upper_bound: Optional[float] = Field(
        default=None,
        description="Upper bound for a stated range or approximate amount, if inferable.",
    )
    is_approximate: Optional[bool] = Field(
        default=None,
        description="True if the transcript uses approximate wording such as about, around, roughly, close to, or a range.",
    )
    currency: str = "USD"
    raw_text: Optional[str] = Field(
        default=None,
        description="Original transcript wording for traceability, e.g. '$165,000' or '43'.",
    )


class PersonalDetails(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    date_of_birth: Optional[str] = Field(
        default=None,
        description="ISO date: YYYY-MM-DD.",
    )
    marital_or_relationship_status: Optional[str] = Field(
        default=None,
        description="Normalized status if mentioned, e.g. married, cohabiting, engaged.",
    )
    mobile_phone: Optional[str] = None
    email: Optional[str] = None


class EmploymentDetails(BaseModel):
    status: Optional[EmploymentStatus] = Field(
        default=None,
        description="Normalized employment status, e.g. employed, self-employed, retired.",
    )
    occupation: Optional[str] = None
    employer: Optional[str] = None
    start_date: Optional[str] = Field(
        default=None,
        description="Employment start date as ISO date or year if only approximate.",
    )
    desired_retirement_age: Optional[int] = None
    desired_retirement_date: Optional[str] = Field(
        default=None,
        description="Desired retirement date as ISO date or year/month if mentioned.",
    )
    tax_residency: Optional[str] = None


class ClientProfile(BaseModel):
    personal: PersonalDetails = Field(default_factory=PersonalDetails)
    employment: EmploymentDetails = Field(default_factory=EmploymentDetails)


class ChildDependant(BaseModel):
    name: Optional[str] = None
    relationship: Optional[str] = Field(
        default=None,
        description="Relationship to client, e.g. child, step-child, parent.",
    )
    date_of_birth: Optional[str] = Field(
        default=None,
        description="ISO date if known; otherwise null.",
    )
    age: Optional[float] = None
    dependency_status: Optional[str] = Field(
        default=None,
        description="Short description such as financially dependent, independent, in childcare.",
    )


class HouseholdDetails(BaseModel):
    partner_or_spouse_name: Optional[str] = Field(
        default=None,
        description="Name of partner/spouse if mentioned. May duplicate client2 if the partner is also treated as a second client.",
    )
    partner_or_spouse_is_client2: Optional[bool] = None
    partner_or_spouse_relationship: Optional[str] = Field(
        default=None,
        description="Relationship status, e.g. spouse, partner, fiancee, cohabiting partner.",
    )
    children_or_dependants: List[ChildDependant] = Field(default_factory=list)


class IncomeItem(BaseModel):
    owner: Owner = Owner.UNKNOWN
    source_name: Optional[str] = Field(
        default=None,
        description="Transcript wording, e.g. current salary, bonus, Social Security.",
    )
    amount: Optional[MoneyAmount] = None
    frequency: Optional[Frequency] = Field(
        default=None,
        description="Payment frequency, e.g. annual, monthly, weekly, one-off.",
    )
    net_or_gross: Optional[NetGross] = Field(
        default=None,
        description="Whether the amount is net, gross, or unknown.",
    )


class ExpenseItem(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description="Normalized spending category, e.g. mortgage, childcare, utilities, travel.",
    )
    owner: Owner = Owner.UNKNOWN
    name: Optional[str] = Field(
        default=None,
        description="Transcript wording or specific expense name.",
    )
    amount: Optional[MoneyAmount] = None
    frequency: Optional[Frequency] = Field(
        default=None,
        description="Payment frequency, e.g. monthly, annual, one-off.",
    )
    is_essential: Optional[bool] = Field(
        default=None,
        description="True for essential expenses; false for discretionary expenses.",
    )


class PensionRetirementItem(BaseModel):
    owner: Owner = Owner.UNKNOWN
    account_type: Optional[str] = Field(
        default=None,
        description="Short normalized product type, e.g. workplace 401(k), IRA, annuity.",
    )
    provider: Optional[str] = None
    current_value: Optional[MoneyAmount] = None
    contribution: Optional[str] = Field(
        default=None,
        description="Contribution amount or percentage if mentioned, preserving the stated unit.",
    )
    has_account_reference_mentioned: Optional[bool] = Field(
        default=None,
        description="True if an account or policy reference was mentioned, but the reference itself is not extracted for privacy.",
    )


class SavingsInvestmentItem(BaseModel):
    owner: Owner = Owner.UNKNOWN
    account_type: Optional[str] = Field(
        default=None,
        description="Short normalized account/product type, e.g. high-yield savings, brokerage, investment fund.",
    )
    provider: Optional[str] = None
    current_value: Optional[MoneyAmount] = None
    cash_value: Optional[MoneyAmount] = Field(
        default=None,
        description="Cash or deposit portion if a split is stated.",
    )
    invested_value: Optional[MoneyAmount] = Field(
        default=None,
        description="Invested portion if a split is stated.",
    )


class LoanMortgageItem(BaseModel):
    owner: Owner = Owner.UNKNOWN
    debt_type: Optional[str] = Field(
        default=None,
        description="Short normalized debt type, e.g. mortgage, car loan, credit card.",
    )
    provider: Optional[str] = None
    monthly_cost: Optional[MoneyAmount] = None
    outstanding_balance: Optional[MoneyAmount] = None
    interest_rate: Optional[float] = Field(
        default=None,
        description="Annual interest rate percentage.",
    )
    no_debt: Optional[bool] = Field(
        default=None,
        description="True if the transcript explicitly states there is no debt in this category.",
    )


class OtherAssetItem(BaseModel):
    owner: Owner = Owner.UNKNOWN
    description: Optional[str] = None
    current_value: Optional[MoneyAmount] = None
    ownership_share: Optional[float] = Field(
        default=None,
        description="Ownership share percentage if mentioned, e.g. 50 for half ownership.",
    )


class ObjectiveItem(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description="Short normalized objective category, e.g. retirement income, tax efficiency, education funding.",
    )
    description: Optional[str] = None
    target_amount_or_income: Optional[MoneyAmount] = None
    target_date_or_age: Optional[str] = Field(
        default=None,
        description="Target date, year, or age if stated.",
    )
    priority: Optional[str] = Field(
        default=None,
        description="Relative priority such as high, medium, or low, if inferable.",
    )
    status_or_uncertainty: Optional[str] = Field(
        default=None,
        description="Current status, uncertainty, or unresolved decision related to the objective.",
    )


class RiskProfilePreferences(BaseModel):
    risk_score_or_label: Optional[str] = Field(
        default=None,
        description="Risk score or label if discussed, e.g. moderate, 5/10.",
    )
    attitude_summary: Optional[str] = None
    preferred_strategy: Optional[str] = Field(
        default=None,
        description="Preferred investment or retirement strategy if discussed.",
    )
    key_concerns: List[str] = Field(default_factory=list)


class EstatePlanning(BaseModel):
    will_status: Optional[str] = Field(
        default=None,
        description="Will status, e.g. has will, no will, considering will, not mentioned.",
    )
    power_of_attorney_status: Optional[str] = Field(
        default=None,
        description="Power of attorney status if mentioned.",
    )
    notes: Optional[str] = None


class CIFExtraction(BaseModel):
    has_client2: Optional[bool] = Field(
        default=None,
        description="True if a second client/partner/spouse is part of the fact find.",
    )
    client1: ClientProfile = Field(default_factory=ClientProfile)
    client2: ClientProfile = Field(default_factory=ClientProfile)
    household: HouseholdDetails = Field(default_factory=HouseholdDetails)
    incomes: List[IncomeItem] = Field(default_factory=list)
    expenses: List[ExpenseItem] = Field(default_factory=list)
    pensions_and_retirement_accounts: List[PensionRetirementItem] = Field(
        default_factory=list,
    )
    savings_and_investments: List[SavingsInvestmentItem] = Field(default_factory=list)
    loans_and_mortgages: List[LoanMortgageItem] = Field(default_factory=list)
    other_assets: List[OtherAssetItem] = Field(default_factory=list)
    objectives: List[ObjectiveItem] = Field(default_factory=list)
    risk_profile_and_preferences: RiskProfilePreferences = Field(
        default_factory=RiskProfilePreferences,
    )
    estate_planning: EstatePlanning = Field(default_factory=EstatePlanning)


# Evaluation helper models


class EvidenceSpan(BaseModel):
    quote: str
    speaker: Optional[str] = None
    timestamp: Optional[str] = None
    line_number: Optional[int] = None


class FieldEvidence(BaseModel):
    field_path: str = Field(
        description="Dotted path such as 'client1.personal.date_of_birth' or 'incomes[0].amount'.",
    )
    evidence: List[EvidenceSpan] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


class LabeledExample(BaseModel):
    example_id: str
    transcript_path: str
    expected: CIFExtraction
    evidence: List[FieldEvidence] = Field(default_factory=list)
