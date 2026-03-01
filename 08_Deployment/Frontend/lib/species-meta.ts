/**
 * Shared species metadata for the BirdSense app.
 * Used by sidebar info cards, frequency highlighting, and similar-species suggestions.
 *
 * ⚠️  AUTO-GENERATED from label_mapping_v3.json — all 87 species match the ML model exactly.
 */

export interface SpeciesMeta {
  name: string
  scientificName: string
  family: string
  habitat: string
  callType: string
  /** Typical vocalization frequency range in kHz */
  freqLow: number
  freqHigh: number
  /** Conservation status */
  status: "common" | "uncommon" | "rare"
}

/**
 * Metadata for all 87 species supported by BirdSense.
 * Keyed by common name (case-sensitive, title case) — must match label_mapping_v3.json english_name exactly.
 */
export const SPECIES_META: Record<string, SpeciesMeta> = {
  // ── Index 30 ──
  "Alder Flycatcher": { name: "Alder Flycatcher", scientificName: "Empidonax alnorum", family: "Tyrannidae", habitat: "Alder thickets, wet shrublands", callType: "Burry 'free-beer!' song", freqLow: 2.0, freqHigh: 7.0, status: "common" },
  // ── Index 55 ──
  "American Avocet": { name: "American Avocet", scientificName: "Recurvirostra americana", family: "Recurvirostridae", habitat: "Shallow wetlands, mudflats", callType: "Loud repeated 'kleek' calls", freqLow: 2.0, freqHigh: 5.0, status: "uncommon" },
  // ── Index 8 ──
  "American Bittern": { name: "American Bittern", scientificName: "Botaurus lentiginosus", family: "Ardeidae", habitat: "Freshwater marshes, wetlands", callType: "Deep booming 'oong-ka-chunk'", freqLow: 0.1, freqHigh: 0.5, status: "uncommon" },
  // ── Index 54 ──
  "American Bushtit": { name: "American Bushtit", scientificName: "Psaltriparus minimus", family: "Aegithalidae", habitat: "Open woodlands, scrub, parks", callType: "Constant soft 'tsrit-tsrit' chattering", freqLow: 5.0, freqHigh: 10.0, status: "common" },
  // ── Index 25 ──
  "American Crow": { name: "American Crow", scientificName: "Corvus brachyrhynchos", family: "Corvidae", habitat: "Widespread — fields, towns", callType: "Harsh 'caw-caw' calls", freqLow: 1.0, freqHigh: 4.0, status: "common" },
  // ── Index 73 ──
  "American Goldfinch": { name: "American Goldfinch", scientificName: "Spinus tristis", family: "Fringillidae", habitat: "Fields, gardens, edges", callType: "Warbling canary-like song", freqLow: 2.0, freqHigh: 8.0, status: "common" },
  // ── Index 32 ──
  "American Kestrel": { name: "American Kestrel", scientificName: "Falco sparverius", family: "Falconidae", habitat: "Open fields, grasslands", callType: "Rapid 'killy-killy-killy'", freqLow: 2.0, freqHigh: 6.0, status: "common" },
  // ── Index 67 ──
  "American Redstart": { name: "American Redstart", scientificName: "Setophaga ruticilla", family: "Parulidae", habitat: "Second-growth deciduous woods", callType: "Variable high 'tse-tse-tse-TSEE'", freqLow: 4.0, freqHigh: 9.0, status: "common" },
  // ── Index 81 ──
  "American Robin": { name: "American Robin", scientificName: "Turdus migratorius", family: "Turdidae", habitat: "Lawns, fields, woodlands", callType: "Caroling 'cheerily-cheeriup'", freqLow: 1.5, freqHigh: 8.0, status: "common" },
  // ── Index 76 ──
  "American Tree Sparrow": { name: "American Tree Sparrow", scientificName: "Spizelloides arborea", family: "Passerellidae", habitat: "Brushy edges, winter fields", callType: "Sweet descending 'teedle-eet'", freqLow: 3.0, freqHigh: 8.0, status: "common" },
  // ── Index 39 ──
  "American Wigeon": { name: "American Wigeon", scientificName: "Mareca americana", family: "Anatidae", habitat: "Marshes, ponds, lakes", callType: "Nasal whistled 'whew-whew'", freqLow: 1.0, freqHigh: 4.0, status: "common" },
  // ── Index 59 ──
  "American Woodcock": { name: "American Woodcock", scientificName: "Scolopax minor", family: "Scolopacidae", habitat: "Wet thickets, young forests", callType: "Nasal buzzy 'peent!' at dusk", freqLow: 1.0, freqHigh: 5.0, status: "uncommon" },
  // ── Index 15 ──
  "Anna's Hummingbird": { name: "Anna's Hummingbird", scientificName: "Calypte anna", family: "Trochilidae", habitat: "Gardens, coastal scrub", callType: "Scratchy buzzy song", freqLow: 2.0, freqHigh: 10.0, status: "common" },
  // ── Index 47 ──
  "Ash-throated Flycatcher": { name: "Ash-throated Flycatcher", scientificName: "Myiarchus cinerascens", family: "Tyrannidae", habitat: "Dry open woodlands, desert scrub", callType: "Soft 'ka-brick' and 'prrt' calls", freqLow: 2.0, freqHigh: 6.0, status: "common" },
  // ── Index 14 ──
  "Baird's Sandpiper": { name: "Baird's Sandpiper", scientificName: "Calidris bairdii", family: "Scolopacidae", habitat: "Grasslands, mudflats, tundra", callType: "Dry rolling 'kreep' call", freqLow: 3.0, freqHigh: 7.0, status: "uncommon" },
  // ── Index 34 ──
  "Bald Eagle": { name: "Bald Eagle", scientificName: "Haliaeetus leucocephalus", family: "Accipitridae", habitat: "Near large water bodies", callType: "High chattering 'kleek-kik-ik-ik'", freqLow: 2.0, freqHigh: 6.0, status: "common" },
  // ── Index 37 ──
  "Baltimore Oriole": { name: "Baltimore Oriole", scientificName: "Icterus galbula", family: "Icteridae", habitat: "Deciduous woodlands", callType: "Rich fluty whistled song", freqLow: 1.5, freqHigh: 5.0, status: "common" },
  // ── Index 35 ──
  "Barn Swallow": { name: "Barn Swallow", scientificName: "Hirundo rustica", family: "Hirundinidae", habitat: "Open areas, farmlands, bridges", callType: "Twittering warble, sharp 'vit-vit'", freqLow: 2.0, freqHigh: 8.0, status: "common" },
  // ── Index 77 ──
  "Barred Owl": { name: "Barred Owl", scientificName: "Strix varia", family: "Strigidae", habitat: "Dense mixed forests, swamps", callType: "'Who cooks for you?' hooting", freqLow: 0.3, freqHigh: 2.0, status: "common" },
  // ── Index 6 ──
  "Bell's Sparrow": { name: "Bell's Sparrow", scientificName: "Artemisiospiza belli", family: "Passerellidae", habitat: "Sagebrush, arid scrublands", callType: "Tinkling musical song", freqLow: 2.5, freqHigh: 7.0, status: "uncommon" },
  // ── Index 40 ──
  "Belted Kingfisher": { name: "Belted Kingfisher", scientificName: "Megaceryle alcyon", family: "Alcedinidae", habitat: "Streams, lakes, coastal areas", callType: "Loud rattling 'kr-r-r-r'", freqLow: 1.0, freqHigh: 6.0, status: "common" },
  // ── Index 78 ──
  "Bewick's Wren": { name: "Bewick's Wren", scientificName: "Thryomanes bewickii", family: "Troglodytidae", habitat: "Scrub, thickets, suburbs", callType: "Complex buzzy trilling song", freqLow: 2.0, freqHigh: 8.0, status: "common" },
  // ── Index 44 ──
  "Black-and-white Warbler": { name: "Black-and-white Warbler", scientificName: "Mniotilta varia", family: "Parulidae", habitat: "Deciduous and mixed forests", callType: "Thin 'weesy-weesy-weesy'", freqLow: 4.0, freqHigh: 9.0, status: "common" },
  // ── Index 22 ──
  "Black-billed Cuckoo": { name: "Black-billed Cuckoo", scientificName: "Coccyzus erythropthalmus", family: "Cuculidae", habitat: "Dense deciduous thickets", callType: "Rapid hollow 'cu-cu-cu-cu'", freqLow: 1.0, freqHigh: 3.5, status: "uncommon" },
  // ── Index 51 ──
  "Black-billed Magpie": { name: "Black-billed Magpie", scientificName: "Pica hudsonia", family: "Corvidae", habitat: "Open woodlands, rangeland, towns", callType: "Harsh nasal 'mag-mag-mag'", freqLow: 1.5, freqHigh: 5.0, status: "common" },
  // ── Index 52 ──
  "Black-capped Chickadee": { name: "Black-capped Chickadee", scientificName: "Poecile atricapillus", family: "Paridae", habitat: "Forests, parks, feeders", callType: "'Chick-a-dee-dee-dee'", freqLow: 2.5, freqHigh: 8.0, status: "common" },
  // ── Index 5 ──
  "Black-chinned Hummingbird": { name: "Black-chinned Hummingbird", scientificName: "Archilochus alexandri", family: "Trochilidae", habitat: "Canyons, gardens, riparian areas", callType: "Soft chase calls, wing buzzing", freqLow: 3.0, freqHigh: 10.0, status: "common" },
  // ── Index 50 ──
  "Black-headed Grosbeak": { name: "Black-headed Grosbeak", scientificName: "Pheucticus melanocephalus", family: "Cardinalidae", habitat: "Open woodlands, riparian areas", callType: "Rich robin-like warbling", freqLow: 1.5, freqHigh: 5.0, status: "common" },
  // ── Index 62 ──
  "Black-throated Blue Warbler": { name: "Black-throated Blue Warbler", scientificName: "Setophaga caerulescens", family: "Parulidae", habitat: "Dense understory of mixed forests", callType: "Buzzy rising 'I'm so lazyyy'", freqLow: 3.0, freqHigh: 7.0, status: "common" },
  // ── Index 69 ──
  "Black-throated Green Warbler": { name: "Black-throated Green Warbler", scientificName: "Setophaga virens", family: "Parulidae", habitat: "Coniferous / mixed forests", callType: "Buzzy 'zoo-zee-zoo-zoo-zee'", freqLow: 3.0, freqHigh: 8.0, status: "common" },
  // ── Index 65 ──
  "Black-throated Grey Warbler": { name: "Black-throated Grey Warbler", scientificName: "Setophaga nigrescens", family: "Parulidae", habitat: "Dry open woodlands, chaparral", callType: "Buzzy rising 'weezy-weezy-weezy-weet'", freqLow: 3.0, freqHigh: 8.0, status: "common" },
  // ── Index 2 ──
  "Black-throated Sparrow": { name: "Black-throated Sparrow", scientificName: "Amphispiza bilineata", family: "Passerellidae", habitat: "Desert scrub, arid hillsides", callType: "Tinkling musical phrases; sharp 'tink'", freqLow: 2.5, freqHigh: 7.0, status: "common" },
  // ── Index 64 ──
  "Blackburnian Warbler": { name: "Blackburnian Warbler", scientificName: "Setophaga fusca", family: "Parulidae", habitat: "Coniferous/mixed forest canopy", callType: "Very high thin 'tsee-tsee-tsee-tseee'", freqLow: 5.0, freqHigh: 10.0, status: "uncommon" },
  // ── Index 68 ──
  "Blackpoll Warbler": { name: "Blackpoll Warbler", scientificName: "Setophaga striata", family: "Parulidae", habitat: "Boreal spruce forests", callType: "Extremely high 'tsi-tsi-tsi-tsi-tsi'", freqLow: 6.0, freqHigh: 10.0, status: "uncommon" },
  // ── Index 49 ──
  "Blue Grosbeak": { name: "Blue Grosbeak", scientificName: "Passerina caerulea", family: "Cardinalidae", habitat: "Hedgerows, brushy edges", callType: "Rich warbling song", freqLow: 2.0, freqHigh: 6.0, status: "uncommon" },
  // ── Index 26 ──
  "Blue Jay": { name: "Blue Jay", scientificName: "Cyanocitta cristata", family: "Corvidae", habitat: "Forests, urban areas", callType: "Loud 'jay-jay', mimicry", freqLow: 1.0, freqHigh: 5.0, status: "common" },
  // ── Index 53 ──
  "Blue-grey Gnatcatcher": { name: "Blue-grey Gnatcatcher", scientificName: "Polioptila caerulea", family: "Polioptilidae", habitat: "Open woodlands, scrub", callType: "Thin wheezy mewing calls", freqLow: 4.0, freqHigh: 9.0, status: "common" },
  // ── Index 84 ──
  "Blue-headed Vireo": { name: "Blue-headed Vireo", scientificName: "Vireo solitarius", family: "Vireonidae", habitat: "Mixed/coniferous forests", callType: "Slow deliberate whistled phrases", freqLow: 2.0, freqHigh: 6.0, status: "common" },
  // ── Index 72 ──
  "Blue-winged Teal": { name: "Blue-winged Teal", scientificName: "Spatula discors", family: "Anatidae", habitat: "Marshes, shallow ponds", callType: "Soft whistled peeping", freqLow: 1.0, freqHigh: 4.0, status: "common" },
  // ── Index 83 ──
  "Blue-winged Warbler": { name: "Blue-winged Warbler", scientificName: "Vermivora cyanoptera", family: "Parulidae", habitat: "Brushy fields, forest edges", callType: "Buzzy 'bee-buzzz' two-note song", freqLow: 3.0, freqHigh: 8.0, status: "uncommon" },
  // ── Index 27 ──
  "Bobolink": { name: "Bobolink", scientificName: "Dolichonyx oryzivorus", family: "Icteridae", habitat: "Tall grass meadows, hayfields", callType: "Bubbly, tinkling flight song", freqLow: 2.0, freqHigh: 8.0, status: "uncommon" },
  // ── Index 21 ──
  "Bonaparte's Gull": { name: "Bonaparte's Gull", scientificName: "Chroicocephalus philadelphia", family: "Laridae", habitat: "Lakes, rivers, coastal waters", callType: "Nasal tern-like chattering", freqLow: 2.0, freqHigh: 5.0, status: "uncommon" },
  // ── Index 31 ──
  "Brewer's Blackbird": { name: "Brewer's Blackbird", scientificName: "Euphagus cyanocephalus", family: "Icteridae", habitat: "Open fields, parking lots, farms", callType: "Creaky squealing 'k-shee'", freqLow: 1.5, freqHigh: 6.0, status: "common" },
  // ── Index 74 ──
  "Brewer's Sparrow": { name: "Brewer's Sparrow", scientificName: "Spizella breweri", family: "Passerellidae", habitat: "Sagebrush plains, arid scrub", callType: "Long rapid buzzy trilling", freqLow: 3.0, freqHigh: 8.0, status: "common" },
  // ── Index 60 ──
  "Broad-tailed Hummingbird": { name: "Broad-tailed Hummingbird", scientificName: "Selasphorus platycercus", family: "Trochilidae", habitat: "Mountain meadows, gardens", callType: "Metallic wing trill in flight", freqLow: 2.0, freqHigh: 10.0, status: "common" },
  // ── Index 13 ──
  "Broad-winged Hawk": { name: "Broad-winged Hawk", scientificName: "Buteo platypterus", family: "Accipitridae", habitat: "Dense deciduous forests", callType: "Thin, high-pitched whistled 'pee-eeee'", freqLow: 2.0, freqHigh: 5.0, status: "common" },
  // ── Index 20 ──
  "Brown Creeper": { name: "Brown Creeper", scientificName: "Certhia americana", family: "Certhiidae", habitat: "Mature forests", callType: "Thin high 'seee' notes", freqLow: 5.0, freqHigh: 10.0, status: "common" },
  // ── Index 80 ──
  "Brown Thrasher": { name: "Brown Thrasher", scientificName: "Toxostoma rufum", family: "Mimidae", habitat: "Dense brush, hedgerows", callType: "Paired musical phrases (repeated twice)", freqLow: 1.5, freqHigh: 8.0, status: "common" },
  // ── Index 45 ──
  "Brown-headed Cowbird": { name: "Brown-headed Cowbird", scientificName: "Molothrus ater", family: "Icteridae", habitat: "Fields, edges, feedlots", callType: "Gurgling 'glug-glug-gleee'", freqLow: 1.5, freqHigh: 6.0, status: "common" },
  // ── Index 4 ──
  "Buff-bellied Pipit": { name: "Buff-bellied Pipit", scientificName: "Anthus rubescens", family: "Motacillidae", habitat: "Tundra, open fields, shorelines", callType: "Thin 'pip-pit' flight call", freqLow: 4.0, freqHigh: 8.0, status: "common" },
  // ── Index 11 ──
  "Bufflehead": { name: "Bufflehead", scientificName: "Bucephala albeola", family: "Anatidae", habitat: "Ponds, lakes, sheltered bays", callType: "Soft growling & squeaky notes", freqLow: 1.0, freqHigh: 4.0, status: "common" },
  // ── Index 36 ──
  "Bullock's Oriole": { name: "Bullock's Oriole", scientificName: "Icterus bullockii", family: "Icteridae", habitat: "Open woodlands, riparian areas", callType: "Whistled chattering song", freqLow: 1.5, freqHigh: 5.0, status: "common" },
  // ── Index 9 ──
  "Canada Goose": { name: "Canada Goose", scientificName: "Branta canadensis", family: "Anatidae", habitat: "Lakes, parks, fields", callType: "Loud honking 'ah-honk'", freqLow: 0.5, freqHigh: 3.0, status: "common" },
  // ── Index 79 ──
  "Carolina Wren": { name: "Carolina Wren", scientificName: "Thryothorus ludovicianus", family: "Troglodytidae", habitat: "Thickets, gardens, undergrowth", callType: "Loud 'teakettle-teakettle'", freqLow: 2.0, freqHigh: 8.0, status: "common" },
  // ── Index 75 ──
  "Chipping Sparrow": { name: "Chipping Sparrow", scientificName: "Spizella passerina", family: "Passerellidae", habitat: "Open woodlands, lawns", callType: "Dry mechanical trill", freqLow: 3.0, freqHigh: 8.0, status: "common" },
  // ── Index 43 ──
  "Common Merganser": { name: "Common Merganser", scientificName: "Mergus merganser", family: "Anatidae", habitat: "Rivers, lakes, large streams", callType: "Harsh croaking calls", freqLow: 0.5, freqHigh: 3.0, status: "common" },
  // ── Index 33 ──
  "Common Yellowthroat": { name: "Common Yellowthroat", scientificName: "Geothlypis trichas", family: "Parulidae", habitat: "Marshes, thickets", callType: "'Witchity-witchity-witchity'", freqLow: 2.0, freqHigh: 7.0, status: "common" },
  // ── Index 0 ──
  "Cooper's Hawk": { name: "Cooper's Hawk", scientificName: "Accipiter cooperii", family: "Accipitridae", habitat: "Forests, wooded neighborhoods", callType: "Rapid 'kek-kek-kek' alarm", freqLow: 2.0, freqHigh: 5.0, status: "common" },
  // ── Index 38 ──
  "Dark-eyed Junco": { name: "Dark-eyed Junco", scientificName: "Junco hyemalis", family: "Passerellidae", habitat: "Forests, feeders, edges", callType: "Musical trill, sharp 'tick'", freqLow: 3.0, freqHigh: 8.0, status: "common" },
  // ── Index 28 ──
  "Downy Woodpecker": { name: "Downy Woodpecker", scientificName: "Dryobates pubescens", family: "Picidae", habitat: "Forests, parks, feeders", callType: "Soft 'pik' call, drumming", freqLow: 1.0, freqHigh: 8.0, status: "common" },
  // ── Index 70 ──
  "Eastern Bluebird": { name: "Eastern Bluebird", scientificName: "Sialia sialis", family: "Turdidae", habitat: "Open woodlands, meadows", callType: "Soft warbling 'tu-a-wee'", freqLow: 2.0, freqHigh: 6.0, status: "common" },
  // ── Index 82 ──
  "Eastern Kingbird": { name: "Eastern Kingbird", scientificName: "Tyrannus tyrannus", family: "Tyrannidae", habitat: "Open areas, meadow edges", callType: "Buzzy sputtering 'dz-dz-dzeee'", freqLow: 2.0, freqHigh: 7.0, status: "common" },
  // ── Index 58 ──
  "Eastern Phoebe": { name: "Eastern Phoebe", scientificName: "Sayornis phoebe", family: "Tyrannidae", habitat: "Woodland edges, bridges", callType: "Raspy 'fee-bee, fee-bee'", freqLow: 2.0, freqHigh: 6.0, status: "common" },
  // ── Index 24 ──
  "Eastern Wood Pewee": { name: "Eastern Wood Pewee", scientificName: "Contopus virens", family: "Tyrannidae", habitat: "Deciduous forests", callType: "Plaintive 'pee-a-wee'", freqLow: 2.0, freqHigh: 6.0, status: "common" },
  // ── Index 10 ──
  "Great Horned Owl": { name: "Great Horned Owl", scientificName: "Bubo virginianus", family: "Strigidae", habitat: "Forests, deserts, urban", callType: "Deep 'hoo-hoo-hoo-hoo'", freqLow: 0.3, freqHigh: 1.5, status: "common" },
  // ── Index 18 ──
  "Hermit Thrush": { name: "Hermit Thrush", scientificName: "Catharus guttatus", family: "Turdidae", habitat: "Forests, understory", callType: "Ethereal flute-like phrases", freqLow: 1.5, freqHigh: 6.0, status: "common" },
  // ── Index 3 ──
  "Mallard": { name: "Mallard", scientificName: "Anas platyrhynchos", family: "Anatidae", habitat: "Ponds, lakes, parks", callType: "Classic 'quack-quack'", freqLow: 0.5, freqHigh: 4.0, status: "common" },
  // ── Index 66 ──
  "Mangrove Warbler": { name: "Mangrove Warbler", scientificName: "Setophaga petechia", family: "Parulidae", habitat: "Mangroves, coastal scrub", callType: "'Sweet-sweet-sweet-so-sweet'", freqLow: 3.0, freqHigh: 8.0, status: "common" },
  // ── Index 41 ──
  "Red-bellied Woodpecker": { name: "Red-bellied Woodpecker", scientificName: "Melanerpes carolinus", family: "Picidae", habitat: "Forests, suburbs", callType: "Rolling 'churr-churr' call", freqLow: 1.0, freqHigh: 5.0, status: "common" },
  // ── Index 85 ──
  "Mourning Dove": { name: "Mourning Dove", scientificName: "Zenaida macroura", family: "Columbidae", habitat: "Open woodlands, suburban", callType: "Mournful 'coo-oo, ooo, oo'", freqLow: 0.3, freqHigh: 1.5, status: "common" },
  // ── Index 63 ──
  "Myrtle Warbler": { name: "Myrtle Warbler", scientificName: "Setophaga coronata", family: "Parulidae", habitat: "Coniferous/mixed forests", callType: "Slow junco-like musical trill", freqLow: 3.0, freqHigh: 8.0, status: "common" },
  // ── Index 17 ──
  "Northern Cardinal": { name: "Northern Cardinal", scientificName: "Cardinalis cardinalis", family: "Cardinalidae", habitat: "Woodlands, gardens", callType: "Loud clear whistles 'cheer-cheer'", freqLow: 2.0, freqHigh: 8.0, status: "common" },
  // ── Index 23 ──
  "Northern Flicker": { name: "Northern Flicker", scientificName: "Colaptes auratus", family: "Picidae", habitat: "Open woodlands, edges", callType: "Loud 'wicka-wicka-wicka'", freqLow: 1.0, freqHigh: 5.0, status: "common" },
  // ── Index 61 ──
  "Northern Parula": { name: "Northern Parula", scientificName: "Setophaga americana", family: "Parulidae", habitat: "Mature forests near water", callType: "Rising buzzy trill 'zeeee-up'", freqLow: 3.0, freqHigh: 9.0, status: "common" },
  // ── Index 29 ──
  "Pileated Woodpecker": { name: "Pileated Woodpecker", scientificName: "Dryocopus pileatus", family: "Picidae", habitat: "Mature forests", callType: "Loud laughing 'kuk-kuk-kuk'", freqLow: 0.8, freqHigh: 4.0, status: "uncommon" },
  // ── Index 48 ──
  "Red Fox Sparrow": { name: "Red Fox Sparrow", scientificName: "Passerella iliaca", family: "Passerellidae", habitat: "Dense thickets, brush piles", callType: "Rich melodious whistled song", freqLow: 2.0, freqHigh: 7.0, status: "common" },
  // ── Index 12 ──
  "Red-tailed Hawk": { name: "Red-tailed Hawk", scientificName: "Buteo jamaicensis", family: "Accipitridae", habitat: "Open country, forests", callType: "Raspy screaming 'keeeeaah'", freqLow: 1.0, freqHigh: 4.0, status: "common" },
  // ── Index 57 ──
  "Black Phoebe": { name: "Black Phoebe", scientificName: "Sayornis nigricans", family: "Tyrannidae", habitat: "Near water, streams, ponds", callType: "Sharp 'tsip' and 'fi-bee' song", freqLow: 2.0, freqHigh: 6.0, status: "common" },
  // ── Index 56 ──
  "Sand Martin": { name: "Sand Martin", scientificName: "Riparia riparia", family: "Hirundinidae", habitat: "Sand banks near water, open areas", callType: "Dry buzzy chattering", freqLow: 2.0, freqHigh: 7.0, status: "common" },
  // ── Index 42 ──
  "Song Sparrow": { name: "Song Sparrow", scientificName: "Melospiza melodia", family: "Passerellidae", habitat: "Marshes, thickets, gardens", callType: "Variable musical trills", freqLow: 2.0, freqHigh: 8.0, status: "common" },
  // ── Index 19 ──
  "Swainson's Thrush": { name: "Swainson's Thrush", scientificName: "Catharus ustulatus", family: "Turdidae", habitat: "Dense forests, migration", callType: "Spiraling upward flute song", freqLow: 1.5, freqHigh: 6.0, status: "common" },
  // ── Index 46 ──
  "Townsend's Solitaire": { name: "Townsend's Solitaire", scientificName: "Myadestes townsendi", family: "Turdidae", habitat: "Mountain coniferous forests", callType: "Clear single-note 'eek' call", freqLow: 2.0, freqHigh: 6.0, status: "uncommon" },
  // ── Index 7 ──
  "Tufted Titmouse": { name: "Tufted Titmouse", scientificName: "Baeolophus bicolor", family: "Paridae", habitat: "Forests, parks, feeders", callType: "Clear 'peter-peter-peter'", freqLow: 2.0, freqHigh: 6.0, status: "common" },
  // ── Index 71 ──
  "White-breasted Nuthatch": { name: "White-breasted Nuthatch", scientificName: "Sitta carolinensis", family: "Sittidae", habitat: "Deciduous forests, feeders", callType: "Nasal 'yank-yank' call", freqLow: 1.5, freqHigh: 6.0, status: "common" },
  // ── Index 86 ──
  "White-throated Sparrow": { name: "White-throated Sparrow", scientificName: "Zonotrichia albicollis", family: "Passerellidae", habitat: "Forest edges, brush", callType: "'Oh Sweet Canada Canada'", freqLow: 2.5, freqHigh: 6.0, status: "common" },
  // ── Index 16 ──
  "Wilson's Warbler": { name: "Wilson's Warbler", scientificName: "Cardellina pusilla", family: "Parulidae", habitat: "Willow thickets, streams", callType: "Rapid chattering song", freqLow: 3.0, freqHigh: 8.0, status: "common" },
  // ── Index 1 ──
  "Wood Duck": { name: "Wood Duck", scientificName: "Aix sponsa", family: "Anatidae", habitat: "Wooded swamps, ponds", callType: "Squealing 'oo-eek!'", freqLow: 1.0, freqHigh: 5.0, status: "common" },
}

/** Get all species names as a sorted array */
export function getSpeciesNames(): string[] {
  return Object.keys(SPECIES_META).sort()
}

/** Get species in same family (excluding given species) */
export function getSimilarSpecies(name: string, limit = 3): SpeciesMeta[] {
  const target = SPECIES_META[name]
  if (!target) return []

  // First try same family
  const sameFamily = Object.values(SPECIES_META)
    .filter((s) => s.family === target.family && s.name !== name)
    .sort((a, b) => {
      // Sort by frequency overlap (closer range = more similar)
      const overlapA = Math.min(a.freqHigh, target.freqHigh) - Math.max(a.freqLow, target.freqLow)
      const overlapB = Math.min(b.freqHigh, target.freqHigh) - Math.max(b.freqLow, target.freqLow)
      return overlapB - overlapA
    })

  if (sameFamily.length >= limit) return sameFamily.slice(0, limit)

  // Fill with frequency-similar species from other families
  const others = Object.values(SPECIES_META)
    .filter((s) => s.name !== name && !sameFamily.includes(s))
    .sort((a, b) => {
      const distA = Math.abs((a.freqLow + a.freqHigh) / 2 - (target.freqLow + target.freqHigh) / 2)
      const distB = Math.abs((b.freqLow + b.freqHigh) / 2 - (target.freqLow + target.freqHigh) / 2)
      return distA - distB
    })

  return [...sameFamily, ...others].slice(0, limit)
}

// ── Recent searches (localStorage) ──

const RECENT_KEY = "birdsense_recent_searches"
const MAX_RECENT = 5

export function getRecentSearches(): string[] {
  if (typeof window === "undefined") return []
  try {
    const raw = localStorage.getItem(RECENT_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

export function addRecentSearch(species: string): void {
  if (typeof window === "undefined") return
  try {
    const current = getRecentSearches().filter((s) => s !== species)
    const updated = [species, ...current].slice(0, MAX_RECENT)
    localStorage.setItem(RECENT_KEY, JSON.stringify(updated))
  } catch {
    // silently fail if localStorage is unavailable
  }
}
