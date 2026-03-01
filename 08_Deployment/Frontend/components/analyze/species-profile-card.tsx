"use client"

import { useState } from "react"
import { MapPin } from "lucide-react"
import { SpeciesMapModal } from "@/components/analyze/species-map-modal"
import { SPECIES_META } from "@/lib/species-meta"

const DEFAULT_BIRD_IMG =
  "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=600&h=400&fit=crop"

/* ── Per-species profile data: conservation, size, migration, behaviour, fun fact ── */
const P: Record<string, [string, string, string, string, string, string, string]> = {
  // [conservation, conservationColor, bodyLength, weight, migratory, behaviour, funFact]
  "Alder Flycatcher": ["Least Concern", "text-green-500", "13 – 17 cm", "12 – 14 g", "Long-distance migrant", "Sits on exposed perches and sallies out to catch insects mid-air.", "Almost identical to Willow Flycatcher — reliably told apart only by voice."],
  "American Avocet": ["Least Concern", "text-green-500", "40 – 51 cm", "275 – 350 g", "Partially migratory", "Sweeps its upturned bill side-to-side through water to catch invertebrates.", "Breeding plumage turns their head and neck a striking rusty-orange color."],
  "American Bittern": ["Least Concern", "text-green-500", "58 – 85 cm", "370 – 500 g", "Partially migratory", "Freezes with bill pointed skyward to mimic marsh reeds when threatened.", "Its booming call can carry over 1 km across marshes and is produced by inflating the esophagus."],
  "American Bushtit": ["Least Concern", "text-green-500", "10 – 11 cm", "5 – 6 g", "Non-migratory", "Travels in active flocks of 10-40 birds, constantly chattering.", "Builds an elaborate hanging sock-shaped nest up to 30 cm long."],
  "American Crow": ["Least Concern", "text-green-500", "40 – 53 cm", "316 – 620 g", "Partially migratory", "Highly social; roosts in winter flocks that can number in the thousands.", "Can recognize individual human faces and hold grudges for years."],
  "American Goldfinch": ["Least Concern", "text-green-500", "11 – 14 cm", "11 – 20 g", "Partially migratory", "Strict vegetarian — one of few songbirds that feed nestlings only seeds.", "Males moult into brilliant yellow breeding plumage each spring."],
  "American Kestrel": ["Least Concern", "text-green-500", "22 – 31 cm", "80 – 165 g", "Partially migratory", "Hovers in place ('kiting') while hunting, scanning the ground for prey.", "Can see ultraviolet light, allowing it to detect rodent urine trails invisible to us."],
  "American Redstart": ["Least Concern", "text-green-500", "11 – 14 cm", "6 – 9 g", "Long-distance migrant", "Fans its tail and wings to startle insects out of hiding, then snatches them.", "Males don't get full black-and-orange plumage until their second year."],
  "American Robin": ["Least Concern", "text-green-500", "23 – 28 cm", "77 – 85 g", "Migratory", "Forages by sight — tilts head to locate earthworms by sound or sight in soil.", "First robin of spring is a well-known cultural sign of the season in North America."],
  "American Tree Sparrow": ["Least Concern", "text-green-500", "14 – 16 cm", "17 – 26 g", "Migratory", "Forages on the ground, often in mixed flocks with juncos in winter.", "Despite its name, it nests on the ground in Arctic tundra — not in trees."],
  "American Wigeon": ["Least Concern", "text-green-500", "42 – 59 cm", "512 – 1,100 g", "Migratory", "Often steals aquatic vegetation brought up by diving ducks and coots.", "Males have a distinctive green eyestripe and a creamy-white forehead."],
  "American Woodcock": ["Least Concern", "text-green-500", "25 – 31 cm", "115 – 280 g", "Migratory", "Performs spectacular spiral courtship flights at dawn and dusk.", "Its eyes are set far back on its head, giving it nearly 360° vision."],
  "Barn Owl": ["Least Concern", "text-green-500", "32 – 40 cm", "224 – 710 g", "Non-migratory", "Hunts almost exclusively by sound using asymmetrical ears.", "Can catch prey in complete darkness using hearing alone — one of nature's best acoustic hunters."],
  "Barn Swallow": ["Least Concern", "text-green-500", "15 – 19 cm", "17 – 20 g", "Long-distance migrant", "Catches insects in continuous acrobatic flight, rarely landing during the day.", "The most widespread swallow species in the world, found on every continent except Antarctica."],
  "Barred Owl": ["Least Concern", "text-green-500", "43 – 50 cm", "470 – 1,050 g", "Non-migratory", "Hunts from perches at night; also wades into shallow water to catch fish.", "Its classic call 'Who cooks for you?' is one of the most recognizable owl sounds."],
  "Bay-breasted Warbler": ["Least Concern", "text-green-500", "12 – 14 cm", "10 – 16 g", "Long-distance migrant", "Deliberately forages in dense spruce canopy, often hanging upside-down.", "Population surges during spruce budworm outbreaks, which provide abundant food."],
  "Belted Kingfisher": ["Least Concern", "text-green-500", "28 – 35 cm", "113 – 178 g", "Partially migratory", "Plunges headfirst into water from hovering flight or perches.", "One of the few bird species where the female is more colorful than the male."],
  "Black Tern": ["Least Concern", "text-green-500", "22 – 26 cm", "50 – 70 g", "Long-distance migrant", "Dips down to pick insects from the water surface in buoyant, erratic flight.", "Unlike most terns, it nests on floating vegetation in freshwater marshes."],
  "Black-and-white Warbler": ["Least Concern", "text-green-500", "12 – 14 cm", "8 – 15 g", "Long-distance migrant", "Creeps along tree trunks and branches like a nuthatch, probing bark.", "One of the earliest warblers to arrive in spring due to its bark-foraging style."],
  "Black-billed Cuckoo": ["Least Concern", "text-green-500", "27 – 31 cm", "40 – 65 g", "Long-distance migrant", "Specializes in eating hairy caterpillars that other birds avoid.", "Periodically sheds its stomach lining to clear accumulated caterpillar hairs."],
  "Black-capped Chickadee": ["Least Concern", "text-green-500", "12 – 15 cm", "9 – 14 g", "Non-migratory", "Caches thousands of food items and can remember each hiding spot.", "Grows new brain neurons each autumn to boost spatial memory for cached food."],
  "Black-throated Blue Warbler": ["Least Concern", "text-green-500", "12 – 14 cm", "8 – 12 g", "Long-distance migrant", "Males and females look so different they were once considered separate species.", "Often forages at lower canopy levels making it easier to observe than most warblers."],
  "Black-throated Sparrow": ["Least Concern", "text-green-500", "12 – 14 cm", "11 – 15 g", "Partially migratory", "Remarkably drought-tolerant; can survive without drinking water for extended periods.", "Gets all the moisture it needs from its insect and seed diet in the desert."],
  "Blackburnian Warbler": ["Least Concern", "text-green-500", "11 – 13 cm", "8 – 13 g", "Long-distance migrant", "Forages high in spruce and hemlock treetops, often hard to see.", "Has the most vivid orange throat of any North American warbler — like a tiny flame."],
  "Blue Jay": ["Least Concern", "text-green-500", "22 – 30 cm", "70 – 100 g", "Partially migratory", "Highly intelligent; known to cache food and use tools.", "Can mimic the calls of Red-tailed Hawks to frighten other birds away from feeders."],
  "Blue-headed Vireo": ["Least Concern", "text-green-500", "12 – 15 cm", "13 – 19 g", "Migratory", "Methodically searches foliage for insects, moving slowly and deliberately.", "Was formerly lumped with Cassin's and Plumbeous Vireos as 'Solitary Vireo'."],
  "Blue-winged Teal": ["Least Concern", "text-green-500", "36 – 41 cm", "270 – 410 g", "Long-distance migrant", "One of the earliest fall migrants among North American ducks.", "Males have a distinctive white crescent on the face and powder-blue shoulders."],
  "Bobolink": ["Least Concern", "text-green-500", "15 – 20 cm", "28 – 56 g", "Long-distance migrant", "Males sing a bubbling, tumbling song during display flights over grasslands.", "Migrates up to 20,000 km round-trip — one of the longest of any songbird."],
  "Brewer's Blackbird": ["Least Concern", "text-green-500", "20 – 25 cm", "50 – 77 g", "Partially migratory", "Walks confidently on the ground in open areas, often near people.", "Males have striking iridescent purple heads and greenish body sheen."],
  "Broad-winged Hawk": ["Least Concern", "text-green-500", "34 – 44 cm", "265 – 560 g", "Long-distance migrant", "Soars on thermals in huge flocks ('kettles') during migration.", "Migration kettles can contain thousands of hawks spiraling in a single thermal."],
  "Brown Creeper": ["Least Concern", "text-green-500", "12 – 14 cm", "7 – 10 g", "Partially migratory", "Spirals up tree trunks probing bark, then flies to the base of the next tree.", "Its cryptic plumage makes it nearly invisible against tree bark."],
  "Brown Thrasher": ["Least Concern", "text-green-500", "23 – 30 cm", "61 – 89 g", "Partially migratory", "Vigorous ground forager; uses its bill to sweep aside leaf litter.", "Has the largest song repertoire of any North American bird — over 1,100 phrases."],
  "Buff-bellied Pipit": ["Least Concern", "text-green-500", "14 – 17 cm", "20 – 28 g", "Migratory", "Bobs its tail constantly while walking on the ground.", "Breeds in alpine tundra and Arctic habitats, then winters in open lowland fields."],
  "Canada Goose": ["Least Concern", "text-green-500", "75 – 110 cm", "2.6 – 6.5 kg", "Partially migratory", "Flies in distinctive V-formation to conserve energy during migration.", "Mate for life and can live over 24 years in the wild."],
  "Canada Warbler": ["Least Concern", "text-green-500", "12 – 15 cm", "9 – 13 g", "Long-distance migrant", "Nests on the ground in dense undergrowth despite being a warbler.", "Has a distinctive 'necklace' of dark streaks across its bright yellow breast."],
  "Cape May Warbler": ["Least Concern", "text-green-500", "12 – 14 cm", "9 – 13 g", "Long-distance migrant", "Has a semi-tubular tongue adapted for sipping nectar and fruit juice.", "Population booms when spruce budworm outbreaks provide abundant caterpillar prey."],
  "Cedar Waxwing": ["Least Concern", "text-green-500", "14 – 17 cm", "30 – 36 g", "Nomadic/irruptive", "Passes berries beak-to-beak along a perched row to share food.", "Waxy red tips on wing feathers get brighter with age, signaling maturity."],
  "Chestnut-sided Warbler": ["Least Concern", "text-green-500", "11 – 14 cm", "8 – 13 g", "Long-distance migrant", "Frequently forages with drooped wings and raised tail in dense thickets.", "Actually benefits from deforestation — more common now than before European settlement."],
  "Clay-colored Sparrow": ["Least Concern", "text-green-500", "12 – 14 cm", "10 – 15 g", "Migratory", "Males sing a flat, insect-like buzzy trill from low bushes.", "One of the plainest sparrows but identified by its distinctive face pattern."],
  "Common Nighthawk": ["Least Concern", "text-green-500", "22 – 25 cm", "55 – 98 g", "Long-distance migrant", "Hunts flying insects at dawn and dusk with mouth wide open in erratic flight.", "Not actually a hawk — it's a nightjar, most closely related to whip-poor-wills."],
  "Common Yellowthroat": ["Least Concern", "text-green-500", "11 – 14 cm", "9 – 12 g", "Migratory", "Skulks in low dense vegetation, often heard before seen.", "Males have a distinctive black mask that varies in size between individuals."],
  "Connecticut Warbler": ["Least Concern", "text-green-500", "13 – 15 cm", "13 – 20 g", "Long-distance migrant", "Walks rather than hops on the ground — unusual for a warbler.", "One of the most elusive warblers; rarely seen despite its loud ringing song."],
  "Cooper's Hawk": ["Least Concern", "text-green-500", "35 – 46 cm", "220 – 680 g", "Partially migratory", "A specialist bird-hunter, agile enough to chase prey through dense forest.", "Has adapted so well to suburbs that backyard bird feeders are now prime hunting spots."],
  "Dark-eyed Junco": ["Least Concern", "text-green-500", "14 – 16 cm", "18 – 30 g", "Partially migratory", "Forages by scratching both feet backward on the ground simultaneously.", "One of North America's most abundant birds with an estimated 630 million individuals."],
  "Eastern Bluebird": ["Least Concern", "text-green-500", "16 – 21 cm", "27 – 34 g", "Partially migratory", "Perches on fence posts and wires, dropping to the ground to catch insects.", "Nearly went extinct due to habitat loss but recovered thanks to nest box programs."],
  "Eastern Kingbird": ["Least Concern", "text-green-500", "19 – 23 cm", "33 – 55 g", "Long-distance migrant", "Fearlessly attacks hawks, crows, and other large birds to defend its territory.", "Has a concealed red-orange crown patch rarely seen except during aggressive displays."],
  "Eastern Meadowlark": ["Near Threatened", "text-yellow-500", "19 – 28 cm", "76 – 150 g", "Partially migratory", "Sings a clear, melodious whistle from fence posts and utility wires.", "Populations have declined over 75% since 1970 due to grassland habitat loss."],
  "Eastern Phoebe": ["Least Concern", "text-green-500", "14 – 17 cm", "16 – 21 g", "Migratory", "Wags its tail constantly — a distinctive field mark visible from a distance.", "Was the first bird ever banded in North America, by John James Audubon in 1804."],
  "Eastern Towhee": ["Least Concern", "text-green-500", "17 – 23 cm", "32 – 53 g", "Partially migratory", "Forages by doing a distinctive two-footed backward scratch in leaf litter.", "Its 'Drink-your-tea!' song is one of the most recognized calls in eastern woodlands."],
  "Eastern Wood-Pewee": ["Least Concern", "text-green-500", "14 – 17 cm", "10 – 19 g", "Long-distance migrant", "Perches on dead branches at mid-canopy, sallying out after flying insects.", "Sings its plaintive 'pee-a-wee' song well into the heat of midday."],
  "Evening Grosbeak": ["Vulnerable", "text-orange-500", "16 – 22 cm", "39 – 86 g", "Irruptive/nomadic", "Crushes large seeds with its massive conical bill.", "Populations have declined over 90% since 1970, making it one of North America's fastest declining birds."],
  "Golden-crowned Kinglet": ["Least Concern", "text-green-500", "8 – 11 cm", "4 – 8 g", "Partially migratory", "Hovers at branch tips to glean tiny insects and eggs from conifer needles.", "Survives sub-zero winter nights by huddling with other kinglets — weighs less than a nickel."],
  "Gray Catbird": ["Least Concern", "text-green-500", "21 – 24 cm", "23 – 56 g", "Migratory", "Mimics the songs of other birds and can sing two notes simultaneously.", "Named for its cat-like mewing call, which it uses as an alarm."],
  "Great Blue Heron": ["Least Concern", "text-green-500", "97 – 137 cm", "1.8 – 3.6 kg", "Partially migratory", "Stands motionless in shallow water, then strikes with lightning speed.", "Largest heron in North America with a wingspan that can exceed 2 meters."],
  "Great Crested Flycatcher": ["Least Concern", "text-green-500", "17 – 21 cm", "27 – 40 g", "Long-distance migrant", "Frequently incorporates shed snakeskins into its nest cavity.", "The snakeskin may deter predators or parasites — scientists are still debating why."],
  "Hairy Woodpecker": ["Least Concern", "text-green-500", "18 – 26 cm", "40 – 95 g", "Non-migratory", "Follows Pileated Woodpeckers to feed on insects exposed in the excavated bark.", "Looks like an oversized Downy Woodpecker — the two are often confused."],
  "House Finch": ["Least Concern", "text-green-500", "13 – 14 cm", "16 – 27 g", "Partially migratory", "Gregarious; travels in roving flocks. Often visits backyard feeders.", "Originally from western North America, introduced to the east in 1940; now widespread."],
  "House Wren": ["Least Concern", "text-green-500", "11 – 13 cm", "10 – 12 g", "Migratory", "Males build multiple 'dummy nests' of sticks; the female picks one to line and use.", "Will puncture the eggs of competing cavity-nesting birds to reduce competition."],
  "Indigo Bunting": ["Least Concern", "text-green-500", "11 – 15 cm", "12 – 18 g", "Long-distance migrant", "Males have no blue pigment — their color comes from light refraction in feather structure.", "Migrates at night using star patterns for navigation, even learning star maps."],
  "Least Flycatcher": ["Least Concern", "text-green-500", "12 – 14 cm", "9 – 13 g", "Long-distance migrant", "Delivers a snappy 'che-BEK!' call repeatedly from an exposed perch.", "Nests in loose colonies, aggressively defending their small territories."],
  "Lincoln's Sparrow": ["Least Concern", "text-green-500", "13 – 15 cm", "15 – 24 g", "Migratory", "Secretive and skulking — prefers to run through dense cover rather than fly.", "Named after Thomas Lincoln, who collected the first specimen with Audubon in 1834."],
  "Magnolia Warbler": ["Least Concern", "text-green-500", "11 – 13 cm", "7 – 13 g", "Long-distance migrant", "Fans its tail while foraging, flashing distinctive white tail patches.", "Named after the magnolia tree where Alexander Wilson first collected it, though it rarely uses magnolias."],
  "Marsh Wren": ["Least Concern", "text-green-500", "10 – 14 cm", "9 – 14 g", "Partially migratory", "Males build up to 22 dummy nests to attract females and confuse predators.", "Sings an incredibly complex song with over 200 different patterns."],
  "Mourning Warbler": ["Least Concern", "text-green-500", "12 – 14 cm", "10 – 15 g", "Long-distance migrant", "Stays low in dense vegetation, walking deliberately along branches.", "Named for the dark hood that looks like mourning crepe on the male's chest."],
  "Nashville Warbler": ["Least Concern", "text-green-500", "11 – 13 cm", "7 – 13 g", "Long-distance migrant", "Bobs its tail while foraging in low brushy habitat.", "Not actually from Nashville — it was named from the location where it was first described."],
  "Northern Cardinal": ["Least Concern", "text-green-500", "21 – 23 cm", "42 – 48 g", "Non-migratory", "Territorial; males sing to defend territory. Females also sing — rare among songbirds.", "One of the few North American songbirds where the female regularly sings."],
  "Northern Flicker": ["Least Concern", "text-green-500", "28 – 36 cm", "86 – 167 g", "Partially migratory", "Unusual woodpecker that feeds primarily on ants on the ground.", "Its tongue can extend 5 cm past the bill tip to extract ants from tunnels."],
  "Northern Waterthrush": ["Least Concern", "text-green-500", "12 – 15 cm", "13 – 25 g", "Long-distance migrant", "Walks along stream edges bobbing its tail constantly like a sandpiper.", "Despite its name, it's a warbler — not a thrush at all."],
  "Olive-sided Flycatcher": ["Near Threatened", "text-yellow-500", "18 – 20 cm", "28 – 40 g", "Long-distance migrant", "Perches conspicuously at the very top of dead trees to scan for insects.", "Its loud 'Quick, three beers!' song carries far and is easy to recognize."],
  "Ovenbird": ["Least Concern", "text-green-500", "14 – 16 cm", "16 – 28 g", "Long-distance migrant", "Walks on the forest floor with a distinctive strut, tail cocked upward.", "Builds a dome-shaped ground nest that resembles a Dutch oven — hence its name."],
  "Palm Warbler": ["Least Concern", "text-green-500", "12 – 14 cm", "7 – 13 g", "Migratory", "Wags its tail constantly — one of the easiest warblers to identify by behaviour.", "One of the few warblers commonly found on the ground rather than in treetops."],
  "Philadelphia Vireo": ["Least Concern", "text-green-500", "11 – 13 cm", "10 – 14 g", "Long-distance migrant", "Sings a song very similar to Red-eyed Vireo but higher-pitched.", "Avoids competition by foraging higher in the canopy than the Red-eyed Vireo."],
  "Pine Warbler": ["Least Concern", "text-green-500", "12 – 14 cm", "9 – 15 g", "Partially migratory", "Almost exclusively found in pine trees, creeping along branches for insects.", "One of the few warblers that regularly eats seeds, allowing it to overwinter."],
  "Purple Finch": ["Least Concern", "text-green-500", "12 – 16 cm", "18 – 32 g", "Partially migratory", "Males look 'dipped in raspberry juice' — a rich wine-red, not actually purple.", "Often confused with House Finch but has a more robust bill and brighter coloring."],
  "Red-breasted Nuthatch": ["Least Concern", "text-green-500", "11 – 12 cm", "8 – 13 g", "Irruptive/nomadic", "Walks headfirst down tree trunks — a unique foraging strategy among birds.", "Smears sticky pine resin around its nest hole entrance to deter predators."],
  "Red-eyed Vireo": ["Least Concern", "text-green-500", "12 – 15 cm", "12 – 26 g", "Long-distance migrant", "Sings persistently from the canopy — one of the most vocal forest birds.", "Holds the record for most songs in a day: one male sang 22,197 songs."],
  "Rose-breasted Grosbeak": ["Least Concern", "text-green-500", "18 – 22 cm", "39 – 49 g", "Long-distance migrant", "Males' song resembles a prettier version of an American Robin's melody.", "Males occasionally sit on the nest — unusual because their bright plumage risks attracting predators."],
  "Ruby-crowned Kinglet": ["Least Concern", "text-green-500", "9 – 11 cm", "5 – 10 g", "Migratory", "Constantly flicks its wings nervously while foraging — a key identification trait.", "Males have a hidden ruby-red crown patch that only shows during excitement or aggression."],
  "Savannah Sparrow": ["Least Concern", "text-green-500", "11 – 17 cm", "15 – 29 g", "Migratory", "Runs through grass rather than flying when disturbed, mouse-like behaviour.", "One of the most widespread sparrows in North America, found in open habitats continent-wide."],
  "Song Sparrow": ["Least Concern", "text-green-500", "12 – 17 cm", "12 – 53 g", "Partially migratory", "One of the most thoroughly studied birds in North America.", "Shows remarkable geographic variation — over 30 subspecies are recognized."],
  "Swainson's Thrush": ["Least Concern", "text-green-500", "16 – 20 cm", "23 – 45 g", "Long-distance migrant", "Sings a beautiful upward-spiraling flute-like song at dusk.", "Migrates at night and can fly nonstop for up to 10 hours."],
  "Swamp Sparrow": ["Least Concern", "text-green-500", "12 – 14 cm", "15 – 23 g", "Partially migratory", "Wades into shallow water to forage — unusual for a sparrow.", "Has longer legs than most sparrows, an adaptation for its wetland lifestyle."],
  "Tennessee Warbler": ["Least Concern", "text-green-500", "11 – 13 cm", "8 – 13 g", "Long-distance migrant", "Population erupts during spruce budworm outbreaks in boreal forests.", "Named from Tennessee where Alexander Wilson first described it, but it only passes through during migration."],
  "Veery": ["Least Concern", "text-green-500", "16 – 19 cm", "26 – 39 g", "Long-distance migrant", "Has one of the most ethereal songs — a downward spiraling 'veer-veer-veer'.", "Can predict hurricane season severity — birds leave early in bad hurricane years."],
  "White-breasted Nuthatch": ["Least Concern", "text-green-500", "13 – 14 cm", "18 – 30 g", "Non-migratory", "Walks headfirst down tree trunks, finding insects other birds miss.", "Sweeps crushed insects around its nest entrance as a chemical deterrent to squirrels."],
  "White-throated Sparrow": ["Least Concern", "text-green-500", "15 – 19 cm", "22 – 32 g", "Migratory", "Famous for its 'Oh sweet Canada Canada Canada' whistle.", "Has two color morphs (tan & white striped) that almost always mate with the opposite morph."],
  "Wilson's Warbler": ["Least Concern", "text-green-500", "10 – 12 cm", "5 – 10 g", "Long-distance migrant", "Active and restless; constantly flits through dense low vegetation.", "Named after Alexander Wilson, the father of American ornithology."],
  "Winter Wren": ["Least Concern", "text-green-500", "8 – 12 cm", "8 – 12 g", "Partially migratory", "Creeps mouse-like through tangles of roots and fallen logs.", "Sings an extraordinarily long and complex song — up to 10 seconds of rapid notes."],
  "Yellow Warbler": ["Least Concern", "text-green-500", "10 – 18 cm", "7 – 25 g", "Long-distance migrant", "Builds a new nest floor on top of cowbird eggs rather than raise the parasite.", "The most widespread warbler in North America, found from Alaska to Mexico."],
  "Yellow-bellied Sapsucker": ["Least Concern", "text-green-500", "19 – 21 cm", "43 – 55 g", "Migratory", "Drills precise rows of small holes in tree bark to create sap 'wells'.", "Over 35 species of birds, mammals, and insects depend on its sap wells for food."],
}

/** Derive conservation color from status string */
function statusColor(s: string): string {
  if (s.startsWith("Least")) return "text-green-500"
  if (s.startsWith("Near")) return "text-yellow-500"
  if (s.startsWith("Vulnerable")) return "text-orange-500"
  return "text-muted-foreground"
}

function getData(species: string) {
  const row = P[species]
  if (row) return {
    img: DEFAULT_BIRD_IMG,
    conservation: row[0], conservationColor: row[1],
    bodyLength: row[2], weight: row[3], migratory: row[4],
    behaviour: row[5], funFact: row[6],
  }
  // Fallback using SPECIES_META
  const meta = SPECIES_META[species]
  if (meta) return {
    img: DEFAULT_BIRD_IMG,
    conservation: meta.status === "rare" ? "Near Threatened" : meta.status === "uncommon" ? "Least Concern" : "Least Concern",
    conservationColor: meta.status === "rare" ? "text-yellow-500" : "text-green-500",
    bodyLength: "—", weight: "—",
    migratory: "Migratory",
    behaviour: `Typically found in ${meta.habitat.toLowerCase()}. Known for its ${meta.callType.toLowerCase()}.`,
    funFact: `Part of the ${meta.family} family. Vocalizes between ${meta.freqLow}–${meta.freqHigh} kHz.`,
  }
  return {
    img: DEFAULT_BIRD_IMG,
    conservation: "Data Deficient", conservationColor: "text-muted-foreground",
    bodyLength: "—", weight: "—", migratory: "Unknown",
    behaviour: "Behaviour data not available.", funFact: "Part of BirdSense's 87-species detection library.",
  }
}

interface SpeciesProfileCardProps {
  species: string
  scientificName: string
}

export function SpeciesProfileCard({ species, scientificName }: SpeciesProfileCardProps) {
  const data = getData(species)
  const [mapOpen, setMapOpen] = useState(false)

  return (
    <div className="border-2 border-foreground border-l-4 border-l-[#22c55e] bg-gradient-to-br from-[#22c55e]/5 to-background">
      <div className="border-b-2 border-foreground px-4 py-2">
        <span className="font-mono text-xs tracking-[0.25em] uppercase text-foreground font-bold">
          SPECIES PROFILE
        </span>
      </div>

      <div className="flex flex-col sm:flex-row">
        {/* Left — Large bird image */}
        <div className="sm:w-[260px] sm:min-w-[260px] h-[240px] sm:h-auto border-b-2 sm:border-b-0 sm:border-r-2 border-foreground overflow-hidden bg-foreground/5 relative">
          <img
            src={data.img}
            alt={species}
            className="w-full h-full object-cover"
          />
          {/* Name overlay */}
          <div className="absolute bottom-0 left-0 right-0 bg-black/70 px-3 py-2">
            <p className="font-mono text-sm font-bold tracking-[0.1em] uppercase text-white">
              {species}
            </p>
            <p className="font-mono text-[10px] tracking-[0.15em] text-white/60 italic">
              {scientificName}
            </p>
          </div>
        </div>

        {/* Right — stats */}
        <div className="flex-1 p-5 space-y-4">
          {/* Conservation badge */}
          <div className="flex items-center gap-2">
            <span className="font-mono text-[11px] tracking-[0.2em] uppercase text-accent w-[90px] shrink-0">
              IUCN STATUS
            </span>
            <span className={`font-mono text-xs font-bold ${data.conservationColor}`}>
              ● {data.conservation}
            </span>
          </div>

          {/* Size / weight / migratory */}
          <div className="grid grid-cols-3 gap-2 border border-foreground/20 divide-x divide-foreground/20">
            <div className="px-3 py-2 text-center">
              <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground block mb-1">LENGTH</span>
              <span className="font-mono text-sm font-bold text-foreground">{data.bodyLength}</span>
            </div>
            <div className="px-3 py-2 text-center">
              <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground block mb-1">WEIGHT</span>
              <span className="font-mono text-sm font-bold text-foreground">{data.weight}</span>
            </div>
            <div className="px-3 py-2 text-center">
              <span className="font-mono text-[10px] tracking-[0.2em] uppercase text-muted-foreground block mb-1">MIGRATION</span>
              <span className="font-mono text-xs font-bold text-foreground leading-tight">{data.migratory}</span>
            </div>
          </div>

          {/* Behaviour */}
          <div>
            <span className="font-mono text-[11px] tracking-[0.2em] uppercase text-accent block mb-1">
              BEHAVIOUR
            </span>
            <p className="font-mono text-xs text-muted-foreground leading-relaxed">
              {data.behaviour}
            </p>
          </div>

          {/* Fun fact */}
          <div className="border-l-2 border-accent pl-3">
            <span className="font-mono text-[11px] tracking-[0.2em] uppercase text-accent block mb-1">
              DID YOU KNOW?
            </span>
            <p className="font-mono text-xs text-muted-foreground leading-relaxed italic">
              {data.funFact}
            </p>
          </div>

          {/* Learn more links */}
          <div className="pt-2 border-t border-foreground/20 space-y-2">
            <span className="font-mono text-[11px] tracking-[0.2em] uppercase text-muted-foreground block">
              LEARN MORE
            </span>
            <div className="flex flex-wrap gap-2">
              {[
                { label: "eBird", icon: "🦅", url: `https://ebird.org/species/${species.replace(/ /g, "_").toLowerCase()}` },
                { label: "Xeno-Canto", icon: "🎵", url: `https://xeno-canto.org/explore?query=${encodeURIComponent(species)}` },
                { label: "Wikipedia", icon: "📖", url: `https://en.wikipedia.org/wiki/${species.replace(/ /g, "_")}` },
                { label: "AllAboutBirds", icon: "🐦", url: `https://www.allaboutbirds.org/guide/${species.replace(/ /g, "_")}` },
              ].map((link) => (
                <a
                  key={link.label}
                  href={link.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1.5 px-2.5 py-1.5 border border-foreground/40 font-mono text-[10px] tracking-[0.15em] uppercase text-muted-foreground hover:text-foreground hover:border-accent cursor-pointer transition-none"
                >
                  <span>{link.icon}</span>
                  {link.label}
                </a>
              ))}
            </div>
          </div>

          {/* Locate button */}
          <div className="flex justify-end pt-2 border-t border-foreground/20">
            <button
              type="button"
              onClick={() => setMapOpen(true)}
              className="flex items-center gap-2 px-4 py-2 border-2 border-accent bg-accent/10 font-mono text-[10px] tracking-[0.2em] uppercase text-accent hover:bg-accent hover:text-white cursor-pointer transition-none font-bold"
            >
              <MapPin size={12} />
              LOCATE ON MAP
            </button>
          </div>
        </div>
      </div>

      {/* Map modal */}
      {mapOpen && (
        <SpeciesMapModal
          species={species}
          scientificName={scientificName}
          onClose={() => setMapOpen(false)}
        />
      )}
    </div>
  )
}
