/**
 * Shared species metadata for the BirdSense app.
 * Used by sidebar info cards, frequency highlighting, and similar-species suggestions.
 *
 * ⚠️  AUTO-GENERATED from label_mapping_combined.json — all 290 species match the ML model exactly.
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
 * Metadata for all 290 species supported by BirdSense.
 * Keyed by common name (case-sensitive, title case) — must match label_mapping_combined.json english_name exactly.
 */
export const SPECIES_META: Record<string, SpeciesMeta> = {
  "Cooper's Hawk": { name: "Cooper's Hawk", scientificName: "Accipiter cooperii", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Wood Duck": { name: "Wood Duck", scientificName: "Aix sponsa", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Black-throated Sparrow": { name: "Black-throated Sparrow", scientificName: "Amphispiza bilineata", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Mallard": { name: "Mallard", scientificName: "Anas platyrhynchos", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Buff-bellied Pipit": { name: "Buff-bellied Pipit", scientificName: "Anthus rubescens", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Black-chinned Hummingbird": { name: "Black-chinned Hummingbird", scientificName: "Archilochus alexandri", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Bell's Sparrow": { name: "Bell's Sparrow", scientificName: "Artemisiospiza belli", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Tufted Titmouse": { name: "Tufted Titmouse", scientificName: "Baeolophus bicolor", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "American Bittern": { name: "American Bittern", scientificName: "Botaurus lentiginosus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Canada Goose": { name: "Canada Goose", scientificName: "Branta canadensis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Great Horned Owl": { name: "Great Horned Owl", scientificName: "Bubo virginianus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Bufflehead": { name: "Bufflehead", scientificName: "Bucephala albeola", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Red-tailed Hawk": { name: "Red-tailed Hawk", scientificName: "Buteo jamaicensis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Broad-winged Hawk": { name: "Broad-winged Hawk", scientificName: "Buteo platypterus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Baird's Sandpiper": { name: "Baird's Sandpiper", scientificName: "Calidris bairdii", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Anna's Hummingbird": { name: "Anna's Hummingbird", scientificName: "Calypte anna", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Wilson's Warbler": { name: "Wilson's Warbler", scientificName: "Cardellina pusilla", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Northern Cardinal": { name: "Northern Cardinal", scientificName: "Cardinalis cardinalis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Hermit Thrush": { name: "Hermit Thrush", scientificName: "Catharus guttatus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Swainson's Thrush": { name: "Swainson's Thrush", scientificName: "Catharus ustulatus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Brown Creeper": { name: "Brown Creeper", scientificName: "Certhia americana", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Bonaparte's Gull": { name: "Bonaparte's Gull", scientificName: "Chroicocephalus philadelphia", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Black-billed Cuckoo": { name: "Black-billed Cuckoo", scientificName: "Coccyzus erythropthalmus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Northern Flicker": { name: "Northern Flicker", scientificName: "Colaptes auratus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Eastern Wood Pewee": { name: "Eastern Wood Pewee", scientificName: "Contopus virens", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "American Crow": { name: "American Crow", scientificName: "Corvus brachyrhynchos", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Blue Jay": { name: "Blue Jay", scientificName: "Cyanocitta cristata", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Bobolink": { name: "Bobolink", scientificName: "Dolichonyx oryzivorus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Downy Woodpecker": { name: "Downy Woodpecker", scientificName: "Dryobates pubescens", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Pileated Woodpecker": { name: "Pileated Woodpecker", scientificName: "Dryocopus pileatus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Alder Flycatcher": { name: "Alder Flycatcher", scientificName: "Empidonax alnorum", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Brewer's Blackbird": { name: "Brewer's Blackbird", scientificName: "Euphagus cyanocephalus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "American Kestrel": { name: "American Kestrel", scientificName: "Falco sparverius", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Common Yellowthroat": { name: "Common Yellowthroat", scientificName: "Geothlypis trichas", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Bald Eagle": { name: "Bald Eagle", scientificName: "Haliaeetus leucocephalus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Barn Swallow": { name: "Barn Swallow", scientificName: "Hirundo rustica", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Bullock's Oriole": { name: "Bullock's Oriole", scientificName: "Icterus bullockii", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Baltimore Oriole": { name: "Baltimore Oriole", scientificName: "Icterus galbula", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Dark-eyed Junco": { name: "Dark-eyed Junco", scientificName: "Junco hyemalis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "American Wigeon": { name: "American Wigeon", scientificName: "Mareca americana", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Belted Kingfisher": { name: "Belted Kingfisher", scientificName: "Megaceryle alcyon", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Red-bellied Woodpecker": { name: "Red-bellied Woodpecker", scientificName: "Melanerpes carolinus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Song Sparrow": { name: "Song Sparrow", scientificName: "Melospiza melodia", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Common Merganser": { name: "Common Merganser", scientificName: "Mergus merganser", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Black-and-white Warbler": { name: "Black-and-white Warbler", scientificName: "Mniotilta varia", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Brown-headed Cowbird": { name: "Brown-headed Cowbird", scientificName: "Molothrus ater", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Townsend's Solitaire": { name: "Townsend's Solitaire", scientificName: "Myadestes townsendi", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Ash-throated Flycatcher": { name: "Ash-throated Flycatcher", scientificName: "Myiarchus cinerascens", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Red Fox Sparrow": { name: "Red Fox Sparrow", scientificName: "Passerella iliaca", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Blue Grosbeak": { name: "Blue Grosbeak", scientificName: "Passerina caerulea", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Black-headed Grosbeak": { name: "Black-headed Grosbeak", scientificName: "Pheucticus melanocephalus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Black-billed Magpie": { name: "Black-billed Magpie", scientificName: "Pica hudsonia", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Black-capped Chickadee": { name: "Black-capped Chickadee", scientificName: "Poecile atricapillus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Blue-grey Gnatcatcher": { name: "Blue-grey Gnatcatcher", scientificName: "Polioptila caerulea", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "American Bushtit": { name: "American Bushtit", scientificName: "Psaltriparus minimus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "American Avocet": { name: "American Avocet", scientificName: "Recurvirostra americana", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Sand Martin": { name: "Sand Martin", scientificName: "Riparia riparia", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Black Phoebe": { name: "Black Phoebe", scientificName: "Sayornis nigricans", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Eastern Phoebe": { name: "Eastern Phoebe", scientificName: "Sayornis phoebe", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "American Woodcock": { name: "American Woodcock", scientificName: "Scolopax minor", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Broad-tailed Hummingbird": { name: "Broad-tailed Hummingbird", scientificName: "Selasphorus platycercus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Northern Parula": { name: "Northern Parula", scientificName: "Setophaga americana", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Black-throated Blue Warbler": { name: "Black-throated Blue Warbler", scientificName: "Setophaga caerulescens", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Myrtle Warbler": { name: "Myrtle Warbler", scientificName: "Setophaga coronata", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Blackburnian Warbler": { name: "Blackburnian Warbler", scientificName: "Setophaga fusca", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Black-throated Grey Warbler": { name: "Black-throated Grey Warbler", scientificName: "Setophaga nigrescens", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Mangrove Warbler": { name: "Mangrove Warbler", scientificName: "Setophaga petechia", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "American Redstart": { name: "American Redstart", scientificName: "Setophaga ruticilla", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Blackpoll Warbler": { name: "Blackpoll Warbler", scientificName: "Setophaga striata", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Black-throated Green Warbler": { name: "Black-throated Green Warbler", scientificName: "Setophaga virens", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Eastern Bluebird": { name: "Eastern Bluebird", scientificName: "Sialia sialis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "White-breasted Nuthatch": { name: "White-breasted Nuthatch", scientificName: "Sitta carolinensis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Blue-winged Teal": { name: "Blue-winged Teal", scientificName: "Spatula discors", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "American Goldfinch": { name: "American Goldfinch", scientificName: "Spinus tristis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Brewer's Sparrow": { name: "Brewer's Sparrow", scientificName: "Spizella breweri", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Chipping Sparrow": { name: "Chipping Sparrow", scientificName: "Spizella passerina", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "American Tree Sparrow": { name: "American Tree Sparrow", scientificName: "Spizelloides arborea", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Barred Owl": { name: "Barred Owl", scientificName: "Strix varia", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Bewick's Wren": { name: "Bewick's Wren", scientificName: "Thryomanes bewickii", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Carolina Wren": { name: "Carolina Wren", scientificName: "Thryothorus ludovicianus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Brown Thrasher": { name: "Brown Thrasher", scientificName: "Toxostoma rufum", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "American Robin": { name: "American Robin", scientificName: "Turdus migratorius", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Eastern Kingbird": { name: "Eastern Kingbird", scientificName: "Tyrannus tyrannus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Blue-winged Warbler": { name: "Blue-winged Warbler", scientificName: "Vermivora cyanoptera", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Blue-headed Vireo": { name: "Blue-headed Vireo", scientificName: "Vireo solitarius", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Mourning Dove": { name: "Mourning Dove", scientificName: "Zenaida macroura", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "White-throated Sparrow": { name: "White-throated Sparrow", scientificName: "Zonotrichia albicollis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Sandhill Crane": { name: "Sandhill Crane", scientificName: "Antigone canadensis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Gambel's Quail": { name: "Gambel's Quail", scientificName: "Callipepla gambelii", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Warbling Vireo": { name: "Warbling Vireo", scientificName: "Vireo gilvus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Common Nighthawk": { name: "Common Nighthawk", scientificName: "Chordeiles minor", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Virginia Rail": { name: "Virginia Rail", scientificName: "Rallus limicola", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Least Flycatcher": { name: "Least Flycatcher", scientificName: "Empidonax minimus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Red-eyed Vireo": { name: "Red-eyed Vireo", scientificName: "Vireo olivaceus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Snow Goose": { name: "Snow Goose", scientificName: "Anser caerulescens", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Sora": { name: "Sora", scientificName: "Porzana carolina", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "California Quail": { name: "California Quail", scientificName: "Callipepla californica", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Common Gallinule": { name: "Common Gallinule", scientificName: "Gallinula galeata", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Eastern Whip-poor-will": { name: "Eastern Whip-poor-will", scientificName: "Antrostomus vociferus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Greater Roadrunner": { name: "Greater Roadrunner", scientificName: "Geococcyx californianus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Northern Bobwhite": { name: "Northern Bobwhite", scientificName: "Colinus virginianus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Wilson's Snipe": { name: "Wilson's Snipe", scientificName: "Gallinago delicata", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Common Poorwill": { name: "Common Poorwill", scientificName: "Phalaenoptilus nuttallii", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Spotted Sandpiper": { name: "Spotted Sandpiper", scientificName: "Actitis macularius", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Wild Turkey": { name: "Wild Turkey", scientificName: "Meleagris gallopavo", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Blue-throated Mountaingem": { name: "Blue-throated Mountaingem", scientificName: "Lampornis clemenciae", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Chuck-will's-widow": { name: "Chuck-will's-widow", scientificName: "Antrostomus carolinensis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Yellow-billed Cuckoo": { name: "Yellow-billed Cuckoo", scientificName: "Coccyzus americanus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "American Coot": { name: "American Coot", scientificName: "Fulica americana", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Pied-billed Grebe": { name: "Pied-billed Grebe", scientificName: "Podilymbus podiceps", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Rivoli's Hummingbird": { name: "Rivoli's Hummingbird", scientificName: "Eugenes fulgens", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Costa's Hummingbird": { name: "Costa's Hummingbird", scientificName: "Calypte costae", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Rufous Hummingbird": { name: "Rufous Hummingbird", scientificName: "Selasphorus rufus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Killdeer": { name: "Killdeer", scientificName: "Charadrius vociferus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Mountain Quail": { name: "Mountain Quail", scientificName: "Oreortyx pictus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Chimney Swift": { name: "Chimney Swift", scientificName: "Chaetura pelagica", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Allen's Hummingbird": { name: "Allen's Hummingbird", scientificName: "Selasphorus sasin", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Cackling Goose": { name: "Cackling Goose", scientificName: "Branta hutchinsii", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Mexican Whip-poor-will": { name: "Mexican Whip-poor-will", scientificName: "Antrostomus arizonae", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Philadelphia Vireo": { name: "Philadelphia Vireo", scientificName: "Vireo philadelphicus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Gadwall": { name: "Gadwall", scientificName: "Mareca strepera", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Willow Flycatcher": { name: "Willow Flycatcher", scientificName: "Empidonax traillii", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Green-winged Teal": { name: "Green-winged Teal", scientificName: "Anas carolinensis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Montezuma Quail": { name: "Montezuma Quail", scientificName: "Cyrtonyx montezumae", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Sharp-tailed Grouse": { name: "Sharp-tailed Grouse", scientificName: "Tympanuchus phasianellus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Least Sandpiper": { name: "Least Sandpiper", scientificName: "Calidris minutilla", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Trumpeter Swan": { name: "Trumpeter Swan", scientificName: "Cygnus buccinator", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "American Black Duck": { name: "American Black Duck", scientificName: "Anas rubripes", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Black Scoter": { name: "Black Scoter", scientificName: "Melanitta americana", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Ring-billed Gull": { name: "Ring-billed Gull", scientificName: "Larus delawarensis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Ring-necked Duck": { name: "Ring-necked Duck", scientificName: "Aythya collaris", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Long-tailed Duck": { name: "Long-tailed Duck", scientificName: "Clangula hyemalis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Hairy Woodpecker": { name: "Hairy Woodpecker", scientificName: "Leuconotopicus villosus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Scaled Quail": { name: "Scaled Quail", scientificName: "Callipepla squamata", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Greater Prairie Chicken": { name: "Greater Prairie Chicken", scientificName: "Tympanuchus cupido", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Broad-billed Hummingbird": { name: "Broad-billed Hummingbird", scientificName: "Cynanthus latirostris", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Short-billed Dowitcher": { name: "Short-billed Dowitcher", scientificName: "Limnodromus griseus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Lesser Yellowlegs": { name: "Lesser Yellowlegs", scientificName: "Tringa flavipes", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Marsh Wren": { name: "Marsh Wren", scientificName: "Cistothorus palustris", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Greater White-fronted Goose": { name: "Greater White-fronted Goose", scientificName: "Anser albifrons", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Northern Shoveler": { name: "Northern Shoveler", scientificName: "Spatula clypeata", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Yellow-bellied Flycatcher": { name: "Yellow-bellied Flycatcher", scientificName: "Empidonax flaviventris", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Ruby-crowned Kinglet": { name: "Ruby-crowned Kinglet", scientificName: "Corthylio calendula", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Ruby-throated Hummingbird": { name: "Ruby-throated Hummingbird", scientificName: "Archilochus colubris", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Solitary Sandpiper": { name: "Solitary Sandpiper", scientificName: "Tringa solitaria", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "White-tailed Ptarmigan": { name: "White-tailed Ptarmigan", scientificName: "Lagopus leucura", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Pacific-slope Flycatcher": { name: "Pacific-slope Flycatcher", scientificName: "Empidonax difficilis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Great Crested Flycatcher": { name: "Great Crested Flycatcher", scientificName: "Myiarchus crinitus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Greater Yellowlegs": { name: "Greater Yellowlegs", scientificName: "Tringa melanoleuca", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Veery": { name: "Veery", scientificName: "Catharus fuscescens", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Lesser Nighthawk": { name: "Lesser Nighthawk", scientificName: "Chordeiles acutipennis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "King Rail": { name: "King Rail", scientificName: "Rallus elegans", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Common Loon": { name: "Common Loon", scientificName: "Gavia immer", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "North American Red Squirrel": { name: "North American Red Squirrel", scientificName: "Tamiasciurus hudsonicus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Black-bellied Whistling Duck": { name: "Black-bellied Whistling Duck", scientificName: "Dendrocygna autumnalis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Hammond's Flycatcher": { name: "Hammond's Flycatcher", scientificName: "Empidonax hammondii", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Ruffed Grouse": { name: "Ruffed Grouse", scientificName: "Bonasa umbellus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Black-necked Stilt": { name: "Black-necked Stilt", scientificName: "Himantopus mexicanus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Grey Catbird": { name: "Grey Catbird", scientificName: "Dumetella carolinensis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Purple Finch": { name: "Purple Finch", scientificName: "Haemorhous purpureus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Olive-sided Flycatcher": { name: "Olive-sided Flycatcher", scientificName: "Contopus cooperi", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Tundra Swan": { name: "Tundra Swan", scientificName: "Cygnus columbianus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Sage Grouse": { name: "Sage Grouse", scientificName: "Centrocercus urophasianus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Yellow-bellied Sapsucker": { name: "Yellow-bellied Sapsucker", scientificName: "Sphyrapicus varius", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Winter Wren": { name: "Winter Wren", scientificName: "Troglodytes hiemalis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "White-throated Swift": { name: "White-throated Swift", scientificName: "Aeronautes saxatalis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Western Grebe": { name: "Western Grebe", scientificName: "Aechmophorus occidentalis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "American Herring Gull": { name: "American Herring Gull", scientificName: "Larus smithsonianus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Ross's Goose": { name: "Ross's Goose", scientificName: "Anser rossii", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Willow Ptarmigan": { name: "Willow Ptarmigan", scientificName: "Lagopus lagopus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "White-winged Dove": { name: "White-winged Dove", scientificName: "Zenaida asiatica", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Merlin": { name: "Merlin", scientificName: "Falco columbarius", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Cassin's Vireo": { name: "Cassin's Vireo", scientificName: "Vireo cassinii", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Golden-crowned Kinglet": { name: "Golden-crowned Kinglet", scientificName: "Regulus satrapa", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Brant Goose": { name: "Brant Goose", scientificName: "Branta bernicla", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Glaucous-winged Gull": { name: "Glaucous-winged Gull", scientificName: "Larus glaucescens", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Common Ostrich": { name: "Common Ostrich", scientificName: "Struthio camelus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Redhead": { name: "Redhead", scientificName: "Aythya americana", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Pine Siskin": { name: "Pine Siskin", scientificName: "Spinus pinus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Northern Saw-whet Owl": { name: "Northern Saw-whet Owl", scientificName: "Aegolius acadicus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Sooty Grouse": { name: "Sooty Grouse", scientificName: "Dendragapus fuliginosus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Limpkin": { name: "Limpkin", scientificName: "Aramus guarauna", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Red-breasted Nuthatch": { name: "Red-breasted Nuthatch", scientificName: "Sitta canadensis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Short-billed Gull": { name: "Short-billed Gull", scientificName: "Larus brachyrhynchus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Ridgway's Rail": { name: "Ridgway's Rail", scientificName: "Rallus obsoletus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Clapper Rail": { name: "Clapper Rail", scientificName: "Rallus crepitans", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Western Wood Pewee": { name: "Western Wood Pewee", scientificName: "Contopus sordidulus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Spring Peeper": { name: "Spring Peeper", scientificName: "Pseudacris crucifer", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Northern Pintail": { name: "Northern Pintail", scientificName: "Anas acuta", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Pauraque": { name: "Pauraque", scientificName: "Nyctidromus albicollis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Calliope Hummingbird": { name: "Calliope Hummingbird", scientificName: "Selasphorus calliope", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Buff-bellied Hummingbird": { name: "Buff-bellied Hummingbird", scientificName: "Amazilia yucatanensis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Black Rail": { name: "Black Rail", scientificName: "Laterallus jamaicensis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Yellow-throated Vireo": { name: "Yellow-throated Vireo", scientificName: "Vireo flavifrons", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Common Pheasant": { name: "Common Pheasant", scientificName: "Phasianus colchicus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Hutton's Vireo": { name: "Hutton's Vireo", scientificName: "Vireo huttoni", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Tree Swallow": { name: "Tree Swallow", scientificName: "Tachycineta bicolor", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Pacific Treefrog": { name: "Pacific Treefrog", scientificName: "Pseudacris regilla", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "American Cliff Swallow": { name: "American Cliff Swallow", scientificName: "Petrochelidon pyrrhonota", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Pacific Wren": { name: "Pacific Wren", scientificName: "Troglodytes pacificus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Clay-colored Sparrow": { name: "Clay-colored Sparrow", scientificName: "Spizella pallida", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Spruce Grouse": { name: "Spruce Grouse", scientificName: "Canachites canadensis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Yellow Rail": { name: "Yellow Rail", scientificName: "Coturnicops noveboracensis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Semipalmated Plover": { name: "Semipalmated Plover", scientificName: "Charadrius semipalmatus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Coyote": { name: "Coyote", scientificName: "Canis latrans", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Surf Scoter": { name: "Surf Scoter", scientificName: "Melanitta perspicillata", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Ruddy Duck": { name: "Ruddy Duck", scientificName: "Oxyura jamaicensis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Dusky Grouse": { name: "Dusky Grouse", scientificName: "Dendragapus obscurus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Grey Partridge": { name: "Grey Partridge", scientificName: "Perdix perdix", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Inca Dove": { name: "Inca Dove", scientificName: "Columbina inca", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Black Tern": { name: "Black Tern", scientificName: "Chlidonias niger", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Cedar Waxwing": { name: "Cedar Waxwing", scientificName: "Bombycilla cedrorum", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Bicknell's Thrush": { name: "Bicknell's Thrush", scientificName: "Catharus bicknelli", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Evening Grosbeak": { name: "Evening Grosbeak", scientificName: "Hesperiphona vespertina", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Common Goldeneye": { name: "Common Goldeneye", scientificName: "Bucephala clangula", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Plain Chachalaca": { name: "Plain Chachalaca", scientificName: "Ortalis vetula", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Band-tailed Pigeon": { name: "Band-tailed Pigeon", scientificName: "Patagioenas fasciata", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "American Three-toed Woodpecker": { name: "American Three-toed Woodpecker", scientificName: "Picoides dorsalis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Douglas's Squirrel": { name: "Douglas's Squirrel", scientificName: "Tamiasciurus douglasii", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Lesser Prairie Chicken": { name: "Lesser Prairie Chicken", scientificName: "Tympanuchus pallidicinctus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Smooth-billed Ani": { name: "Smooth-billed Ani", scientificName: "Crotophaga ani", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Eurasian Collared Dove": { name: "Eurasian Collared Dove", scientificName: "Streptopelia decaocto", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Horned Grebe": { name: "Horned Grebe", scientificName: "Podiceps auritus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Hooded Merganser": { name: "Hooded Merganser", scientificName: "Lophodytes cucullatus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Great Blue Heron": { name: "Great Blue Heron", scientificName: "Ardea herodias", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Allardâ€™s ground cricket": { name: "Allardâ€™s ground cricket", scientificName: "Allonemobius allardi", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Red-breasted Sapsucker": { name: "Red-breasted Sapsucker", scientificName: "Sphyrapicus ruber", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Black-backed Woodpecker": { name: "Black-backed Woodpecker", scientificName: "Picoides arcticus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Fulvous Whistling Duck": { name: "Fulvous Whistling Duck", scientificName: "Dendrocygna bicolor", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Buff-collared Nightjar": { name: "Buff-collared Nightjar", scientificName: "Antrostomus ridgwayi", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Hudsonian Godwit": { name: "Hudsonian Godwit", scientificName: "Limosa haemastica", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Steller's Jay": { name: "Steller's Jay", scientificName: "Cyanocitta stelleri", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Boreal Chickadee": { name: "Boreal Chickadee", scientificName: "Poecile hudsonicus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Vaux's Swift": { name: "Vaux's Swift", scientificName: "Chaetura vauxi", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Clark's Grebe": { name: "Clark's Grebe", scientificName: "Aechmophorus clarkii", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Western Osprey": { name: "Western Osprey", scientificName: "Pandion haliaetus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Wood Thrush": { name: "Wood Thrush", scientificName: "Hylocichla mustelina", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Barrow's Goldeneye": { name: "Barrow's Goldeneye", scientificName: "Bucephala islandica", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Lucifer Sheartail": { name: "Lucifer Sheartail", scientificName: "Calothorax lucifer", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Violet-crowned Hummingbird": { name: "Violet-crowned Hummingbird", scientificName: "Leucolia violiceps", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Groove-billed Ani": { name: "Groove-billed Ani", scientificName: "Crotophaga sulcirostris", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Common Ground Dove": { name: "Common Ground Dove", scientificName: "Columbina passerina", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "House Wren": { name: "House Wren", scientificName: "Troglodytes aedon", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Mottled Duck": { name: "Mottled Duck", scientificName: "Anas fulvigula", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Mexican Duck": { name: "Mexican Duck", scientificName: "Anas diazi", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Rock Ptarmigan": { name: "Rock Ptarmigan", scientificName: "Lagopus muta", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Black Oystercatcher": { name: "Black Oystercatcher", scientificName: "Haematopus bachmani", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Green Heron": { name: "Green Heron", scientificName: "Butorides virescens", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Varied Thrush": { name: "Varied Thrush", scientificName: "Ixoreus naevius", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Pine Grosbeak": { name: "Pine Grosbeak", scientificName: "Pinicola enucleator", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Spectacled Eider": { name: "Spectacled Eider", scientificName: "Somateria fischeri", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Red-necked Grebe": { name: "Red-necked Grebe", scientificName: "Podiceps grisegena", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Common Tern": { name: "Common Tern", scientificName: "Sterna hirundo", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Black-crowned Night Heron": { name: "Black-crowned Night Heron", scientificName: "Nycticorax nycticorax", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "American Dusky Flycatcher": { name: "American Dusky Flycatcher", scientificName: "Empidonax oberholseri", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Northern Raven": { name: "Northern Raven", scientificName: "Corvus corax", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Nene": { name: "Nene", scientificName: "Branta sandvicensis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Cinnamon Teal": { name: "Cinnamon Teal", scientificName: "Spatula cyanoptera", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Indian Peafowl": { name: "Indian Peafowl", scientificName: "Pavo cristatus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Common Eider": { name: "Common Eider", scientificName: "Somateria mollissima", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Marbled Godwit": { name: "Marbled Godwit", scientificName: "Limosa fedoa", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Willet": { name: "Willet", scientificName: "Tringa semipalmata", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "House Finch": { name: "House Finch", scientificName: "Haemorhous mexicanus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Two-barred Crossbill": { name: "Two-barred Crossbill", scientificName: "Loxia leucoptera", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Smith's Longspur": { name: "Smith's Longspur", scientificName: "Calcarius pictus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "American Toad": { name: "American Toad", scientificName: "Anaxyrus americanus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Egyptian Goose": { name: "Egyptian Goose", scientificName: "Alopochen aegyptiaca", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Chukar Partridge": { name: "Chukar Partridge", scientificName: "Alectoris chukar", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Berylline Hummingbird": { name: "Berylline Hummingbird", scientificName: "Saucerottia beryllina", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "White-tipped Dove": { name: "White-tipped Dove", scientificName: "Leptotila verreauxi", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Purple Gallinule": { name: "Purple Gallinule", scientificName: "Porphyrio martinica", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Hudsonian Whimbrel": { name: "Hudsonian Whimbrel", scientificName: "Numenius hudsonicus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Northern Harrier": { name: "Northern Harrier", scientificName: "Circus hudsonius", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Red-naped Sapsucker": { name: "Red-naped Sapsucker", scientificName: "Sphyrapicus nuchalis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Mountain Chickadee": { name: "Mountain Chickadee", scientificName: "Poecile gambeli", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Purple Martin": { name: "Purple Martin", scientificName: "Progne subis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Fall Field Cricket": { name: "Fall Field Cricket", scientificName: "Gryllus pennsylvanicus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Pig Frog": { name: "Pig Frog", scientificName: "Aquarana grylio", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "King Eider": { name: "King Eider", scientificName: "Somateria spectabilis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Red-breasted Merganser": { name: "Red-breasted Merganser", scientificName: "Mergus serrator", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Hawaiian Coot": { name: "Hawaiian Coot", scientificName: "Fulica alai", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Upland Sandpiper": { name: "Upland Sandpiper", scientificName: "Bartramia longicauda", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Razorbill": { name: "Razorbill", scientificName: "Alca torda", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Least Bittern": { name: "Least Bittern", scientificName: "Ixobrychus exilis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Northern Goshawk": { name: "Northern Goshawk", scientificName: "Accipiter gentilis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Acadian Flycatcher": { name: "Acadian Flycatcher", scientificName: "Empidonax virescens", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Canada Jay": { name: "Canada Jay", scientificName: "Perisoreus canadensis", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Sedge Wren": { name: "Sedge Wren", scientificName: "Cistothorus stellaris", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Mute Swan": { name: "Mute Swan", scientificName: "Cygnus olor", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 },
  "Harlequin Duck": { name: "Harlequin Duck", scientificName: "Histrionicus histrionicus", family: "Unknown", habitat: "Various", callType: "Vocalization", status: "common", freqLow: 2.0, freqHigh: 8.0 }
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
