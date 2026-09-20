# DSP–DSI join to DSLD labels

Checked the local compiled CSVs and raw DSLD records on 2026-09-19.

`compiled/dsp_dsi_joined.csv` already contains `dsp_dsld_id`. Joining
`dsp_id` to `dsp.csv:iDISK_ID` confirmed that all 317,062 existing ID cells
exactly match `dsp.csv:DSLD_ID`. There are no missing product keys and the
163,806 product keys are unique.

Created [dsp_dsi_joined_with_dsld.csv](../datasets/iDISK2.0/compiled/dsp_dsi_joined_with_dsld.csv)
with an additional scalar `dsld_id` column. Pipe-separated product mappings
are expanded to one row per ingredient edge and label ID. Products without
a DSLD mapping retain one row with an empty `dsld_id`.

| Count | Value |
|---|---:|
| Original ingredient edges | 317,062 |
| Original edges with DSLD mappings | 2,156 |
| Output rows after label expansion | 319,292 |
| Output rows with a scalar DSLD ID | 4,386 |
| Products represented by those DSLD mappings | 728 |
| Unique DSLD label IDs represented | 1,477 |

Every output row was read back and compared with its original ingredient
edge and product mapping. Original CSVs are unchanged. All original columns,
including `evidence_source` and the full `dsp_dsld_id` list, are retained.

## Ingredient spot checks

Manually compared 13 local labels belonging to eight selected products.
Inspection included nested ingredient rows, forms, ingredient groups, notes,
and the DSI entity's source names. These are selected examples, not a random
sample or an estimate of overall accuracy.

| Product / iDISK ID | DSLD labels checked | Joined ingredient(s) | Finding |
|---|---|---|---|
| Moringa / DSP000641 | 34066, 73755 | Moringa oleifera | Corresponds to the sole supplement row, Moringa oleifera leaf powder, on both labels. Relationship source is NHP. |
| Calcium Malate / DSP002772 | 276112 | Vitamin D; Calcium | Both correspond to the two supplement rows. Forms specify cholecalciferol and di-calcium malate. Relationship source is NHP. |
| Ascorbic Acid Capsules / DSP005172 | 218944 | Vitamin C | Corresponds to the sole supplement row, vitamin C as ascorbic acid. Relationship source is NHP. |
| Magnesium (Citrate) / DSP005237 | 184625 | Magnesium | Corresponds to the sole supplement row, magnesium as magnesium citrate. Relationship source is NHP. |
| Resveratrol 500 mg / DSP071598 | 795 | Grape | Supported as a source/group of the red-wine polyphenol extract, but incomplete: Japanese knotweed extract and quercetin also appear. Relationship source is DSLD. |
| Herba Vision / DSP071735 | 939, 20999 | Zeaxanthin | Supported in a lutein note on 939 and a nested lutein row on 20999. Incomplete: the main supplement rows are lutein and bilberry extract. Relationship source is DSLD. |
| Cocoa / DSP071996 | 1242 | Flavanol | Corresponds to a nested total-flavanols declaration. Incomplete: cocoa powder, green tea, acacia catechu, EGCG, catechin and epicatechin also appear. Relationship source is DSLD. |
| Best L-Tryptophan 500 mg / DSP072012 | 1258, 24588, 25209, 25210 | 3-hydroxy-5-(hydroxymethyl)-2-methylisonicotinaldehyde 5-phosphate | The DSI source material identifies this as pyridoxal 5-phosphate, a vitamin B6 form. It appears under vitamin B6 on label 25210. The other three labels list only L-tryptophan in their supplement rows. All four contain L-tryptophan, which is absent from the joined ingredient set. Label 25210 also lists niacin. Relationship source is DSLD. |

The product ID join is correct, but ingredient relationships are not always
complete or valid for every label version attached to a product. Expanding
IDs creates candidate label/ingredient pairs; it does not establish
label-specific ingredient evidence. Keep `evidence_source`: a DSLD product
mapping can coexist with an NHP ingredient relationship.

## Reproduction and evidence

From the repository root:

```sh
python3 datasets/iDISK2.0/compiled/build_dsp_dsi_joined_with_dsld.py
```

The [audit JSON](../datasets/iDISK2.0/compiled/dsp_dsi_joined_with_dsld.audit.json)
contains counts, input/output checksums, the selected joined ingredients with
DSI source names, and raw ingredient rows and other ingredients for all 13
labels. Each raw label has a source path and checksum. No network or model
calls are needed. The script and output CSVs are in the ignored local dataset
directory; this report is outside it.
