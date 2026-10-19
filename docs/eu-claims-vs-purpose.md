# EU health claims compared with Purpose

**Status:** observation, no mapping or integration done
**Date:** 2026-10-14
**Data:** [`datasets/eu_claims/health_claims_official.csv`](../datasets/eu_claims/health_claims_official.csv)

Do the claims in the EU Register of nutrition and health claims look like
iDISK's `Purpose`, and like the purpose claims we extract from DSLD labels?
Short answer: they are the same kind of statement, a benefit or function claim
rather than a warning. They differ in who the claim is about, in wording, and
in carrying a regulator's verdict.

iDISK is the reference: if the register is integrated, its substances map to
iDISK **DSI** nodes. That mapping has not been started.

---

## 1. The register in numbers

| | count |
|---|---|
| rows | 3,342 |
| distinct claims (`policy_item_code`) | 2,337 |
| Authorised / Non-authorised / Revoked rows | 1,022 / 2,316 / 4 |
| distinct substances (`nutrient_subst_food_no_html`) | 1,229 |

| `claim_type` | Authorised | Non-authorised | Revoked |
|---|---|---|---|
| Art.13(1): general function | 945 | 1,957 | 4 |
| Art.13(5): based on new science | 19 | 212 | |
| Art.14(1)(a): disease-risk reduction | 43 | 57 | |
| Art.14(1)(b): children's development | 15 | 90 | |

**A row is not a claim.** 382 claim codes have more than one row. Rows with the
same code differ only in `efsa_question` / `efsa_question_url`, `entry_id`,
`reasons_of_non_authorisation` and the legislation columns. So a row is
*claim × EFSA opinion*, and one claim should be keyed by `policy_item_code`,
with its rows kept as evidence. No two rows are exact duplicates.

**Data quality to handle before use:**

- `nutrient_subst_food_no_html` still contains `</em>`, `®`, `\x99` mojibake and
  `\xa0`.
- Some Non-authorised `claim` cells hold several bullet claims (`* …`), the
  submission form's labels ("Exact wording of claim as it appears on
  product: …"), or text in another language (Italian).
- `health_relation` is `"not validated"` on 389 rows, all Non-authorised, and
  empty on 433.
- `policy_item_id` is a float string (`755369.0`); dates are `dd/mm/yyyy`;
  `legislation_pub_date`, `legislation_entry_into_force_date` and
  `legislation_application_date` are empty on every row.
- The folder has no README, download date or fetch script, unlike
  `datasets/au_tga/`.

## 2. Samples side by side

Random samples (Python `random`, seed 1 for iDISK and EU, seed 2 for DSLD);
"…" marks where a sample was cut off.

**EU Authorised** (`substance | claim`):

- Selenium | Selenium contributes to the normal function of the immune system
- Vitamin B12 | Vitamin B12 contributes to normal functioning of the nervous system
- Choline | Choline contributes to the maintenance of normal liver function
- Biotin | Biotin contributes to normal macronutrient metabolism
- Vitamin D | Vitamin D has a role in the process of cell division
- Vitamin C | Vitamin C contributes to normal collagen formation for the normal function of gums
- Glucomannan (konjac mannan) | Glucomannan contributes to the maintenance of normal blood cholesterol levels
- Meal replacement for weight control | Substituting two daily meals of an energy restricted diet with meal replacements contributes to weight loss
- Plant sterols | Plant sterols have been shown to lower/reduce blood cholesterol. High cholesterol is a risk factor in the development of coronary heart disease.

**EU Non-authorised:**

- Vitamin B6 | Necessary for healthy teeth, bones, hair, skin and nails.
- Hydrolised guar gum | Improves health bowel/helps promote regularity
- Bilberry and anthocyanins from blackcurr… | The anthocyanins from bilberry and mountain blackcurrant are good for the eyes (improve adaptation to the dark and relieve eye tiredness caused by computer terminal work)
- I omega kids®/Pufan 3 kids® … | Help to promote concentration
- OPC Premium, containing 40 mg oligomeri… | OPC have been shown to reduce blood cholesterol levels and may therefore reduce the risk of cardiovascular disease
- Ocean Spray Cranberry Products® … | Regular consumption of 2 servings per day of an Ocean Spray product each containing typically 80 mg cranberry proanthocyanidins helps reduce the risk of urinary tract inf…
- Lactobacillus paracasei B21060 | Favorisce il riequilibrio dell'ecosistema intestinale Clarification provided - contributes to the rebalancing of intestinal microflora; …
- Name of Food product: … | Exact wording of claim as it appears on product: Seaweed fibres support body detoxification …

**iDISK `Purpose`** (LNHPD, one cell per product; 65,097 filled):

- A multi-vitamin/mineral supplement that helps in the maintenance of good health. It also helps in the development and maintenance of bones, cartilage, teeth and gums.
- A factor in the maintenance of good health. Helps the body to metabolize carbohydrates, proteins and fats. Helps to form red blood cells. Helps to prevent vitamin B12 deficiency.
- Helps to enhance cognitive function in adults. Helps to enhance memory in adults. Helps to support peripheral circulation.
- Helps in the maintenance of bones and teeth. Helps to prevent bone resorption and osteoporosis. A factor in the maintenance of good health.
- (Traditionally) used in Herbal Medicine to aid digestion. Traditionally used in Herbal Medicine to help relieve headaches.
- Homeopathic medicine used for the temporary relief from symptoms of dyspepsia such as flatulence and nausea.
- Product manufactured based on Traditional Chinese Medicine theory.

**Our DSLD purpose claims** (run `gold1000-r4-2610130950-0`, 2,354 claims):

- biotin supports carbohydrate protein fat metabolism
- clinical studies indicate that lycopene helps to support cardiovascular health and normal immune system function
- maintain bone muscle and connective tissue health
- immune system support
- antioxidant support
- maintain healthy skin hair and nails
- increase energy
- non drowsy
- reclaim your birthright to radical health

## 3. Observations

| | EU Authorised | EU Non-authorised | iDISK Purpose | DSLD purpose |
|---|---|---|---|---|
| about | one substance | one substance or a branded product | a product | a product |
| wording | fixed: "X contributes to normal Y" | the applicant's own | Health Canada monograph wording | the label's own |
| claims per item | one | one, sometimes a bullet list | several in one cell | one per row |
| verdict | accepted | rejected by EFSA | licensed product | none |

1. **Same kind of statement.** All four are benefit or function claims. None
   is a warning.
2. **EU Authorised is closest to iDISK Purpose in tone.** Both use regulator
   wording ("contributes to…", "helps in the maintenance of…"). But iDISK Purpose
   belongs to a product and packs several claims into one cell. It also has
   traditional-use, homeopathic and TCM indications, which the EU register has
   no equivalent for.
3. **EU Non-authorised is closest to DSLD purpose in style.** It is wording
   companies wanted to use, so it reads like a label: "good for the eyes",
   "helps promote regularity".
4. **EU claims are about a substance, which fits DSI directly.** Purpose has to
   go through product → ingredient (spec 0009); an EU claim already names its
   substance.
5. **EU claims carry a verdict.** Purpose records what is claimed; the register
   records whether EFSA accepted it. A Non-authorised claim means EFSA found no
   cause-and-effect. It is evidence about the claim and must not become an
   effectiveness edge.
6. **Authorised claims can be narrower than label wording.** "Normal collagen
   formation for the normal function of gums" against a label's "supports
   healthy skin".
7. **Not everything in the register is purpose-like:**
   - **Art.14(1)(a) claims are disease-risk reduction** ("…a risk factor in the
     development of coronary heart disease"). They are closer to iDISK's
     DSI → disease `is_effective_for` than to Purpose.
   - **Some claims are about foods or diets**, not ingredients (meal
     replacements, sugar-free chewing gum, foods low in sodium).
   - **`restrictions_of_use` is risk-like**: "The claim shall not be used for
     foods targeting children.", choking warnings. Only 81 rows fill it.
   - **`condition_of_use` sets a dose threshold** ("…food which contains at
     least 40 mg of DHA per 100 g…"). Neither Purpose nor Risk has this.

## 4. What this suggests

- **EU Authorised claims can serve as approved reference wording for a DSI's
  purpose.**
- **EU Non-authorised claims can serve as rejected purposes**: real-world
  wording with a verdict attached.
- **They could be compared against our purpose claims later.** The wording is
  close enough to DSLD label text that, once claims are attributed to
  ingredients, an embedding comparison could tag a label claim as close to an
  approved EU claim, close to a rejected one, or neither. Not designed yet.
