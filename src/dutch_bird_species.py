#!/usr/bin/env python3
"""
Complete lijst van vogelsoorten die in Nederland voorkomen.
Gebaseerd op de IOC World Bird List en Sovon waarnemingen.

Bevat:
- Nederlandse naam
- Wetenschappelijke naam (voor Xeno-canto)
- Directory naam (voor bestanden)
- Prioriteit (1=algemeen, 2=regelmatig, 3=zeldzaam)
"""

# Format: (Nederlandse naam, Wetenschappelijke naam, directory_naam, prioriteit)
# Prioriteit 1: Zeer algemeen - focus voor eerste training
# Prioriteit 2: Regelmatig - tweede batch
# Prioriteit 3: Minder algemeen - derde batch

DUTCH_BIRD_SPECIES = [
    # ============================================================
    # PRIORITEIT 1: ZEER ALGEMENE SOORTEN
    # ============================================================

    # Mezen
    ("Koolmees", "Parus major", "koolmees", 1),
    ("Pimpelmees", "Cyanistes caeruleus", "pimpelmees", 1),
    ("Staartmees", "Aegithalos caudatus", "staartmees", 1),
    ("Kuifmees", "Lophophanes cristatus", "kuifmees", 2),
    ("Zwarte Mees", "Periparus ater", "zwarte_mees", 2),
    ("Glanskop", "Poecile palustris", "glanskop", 2),
    ("Matkop", "Poecile montanus", "matkop", 2),

    # Lijsters
    ("Merel", "Turdus merula", "merel", 1),
    ("Zanglijster", "Turdus philomelos", "zanglijster", 1),
    ("Grote Lijster", "Turdus viscivorus", "grote_lijster", 2),
    ("Koperwiek", "Turdus iliacus", "koperwiek", 1),
    ("Kramsvogel", "Turdus pilaris", "kramsvogel", 2),
    ("Beflijster", "Turdus torquatus", "beflijster", 3),

    # Roodborstachtigen
    ("Roodborst", "Erithacus rubecula", "roodborst", 1),
    ("Blauwborst", "Luscinia svecica", "blauwborst", 2),
    ("Nachtegaal", "Luscinia megarhynchos", "nachtegaal", 2),
    ("Zwarte Roodstaart", "Phoenicurus ochruros", "zwarte_roodstaart", 2),
    ("Gekraagde Roodstaart", "Phoenicurus phoenicurus", "gekraagde_roodstaart", 2),

    # Vinken
    ("Vink", "Fringilla coelebs", "vink", 1),
    ("Keep", "Fringilla montifringilla", "keep", 2),
    ("Groenling", "Chloris chloris", "groenling", 1),
    ("Putter", "Carduelis carduelis", "putter", 1),
    ("Sijs", "Spinus spinus", "sijs", 2),
    ("Kneu", "Linaria cannabina", "kneu", 2),
    ("Barmsijs", "Acanthis flammea", "barmsijs", 3),
    ("Kruisbek", "Loxia curvirostra", "kruisbek", 3),
    ("Goudvink", "Pyrrhula pyrrhula", "goudvink", 2),
    ("Appelvink", "Coccothraustes coccothraustes", "appelvink", 2),

    # Mussen
    ("Huismus", "Passer domesticus", "huismus", 1),
    ("Ringmus", "Passer montanus", "ringmus", 2),
    ("Heggenmus", "Prunella modularis", "heggenmus", 1),

    # Gorzen
    ("Rietgors", "Emberiza schoeniclus", "rietgors", 2),
    ("Geelgors", "Emberiza citrinella", "geelgors", 2),
    ("Grauwe Gors", "Emberiza calandra", "grauwe_gors", 3),

    # Kwikstaarten en Piepers
    ("Witte Kwikstaart", "Motacilla alba", "witte_kwikstaart", 1),
    ("Gele Kwikstaart", "Motacilla flava", "gele_kwikstaart", 2),
    ("Grote Gele Kwikstaart", "Motacilla cinerea", "grote_gele_kwikstaart", 2),
    ("Graspieper", "Anthus pratensis", "graspieper", 2),
    ("Boompieper", "Anthus trivialis", "boompieper", 2),
    ("Waterpieper", "Anthus spinoletta", "waterpieper", 3),
    ("Oeverpieper", "Anthus petrosus", "oeverpieper", 3),

    # Zangers
    ("Winterkoning", "Troglodytes troglodytes", "winterkoning", 1),
    ("Tjiftjaf", "Phylloscopus collybita", "tjiftjaf", 1),
    ("Fitis", "Phylloscopus trochilus", "fitis", 1),
    ("Fluiter", "Phylloscopus sibilatrix", "fluiter", 2),
    ("Zwartkop", "Sylvia atricapilla", "zwartkop", 1),
    ("Tuinfluiter", "Sylvia borin", "tuinfluiter", 2),
    ("Grasmus", "Curruca communis", "grasmus", 2),
    ("Braamsluiper", "Curruca curruca", "braamsluiper", 2),
    ("Spotvogel", "Hippolais icterina", "spotvogel", 2),
    ("Kleine Karekiet", "Acrocephalus scirpaceus", "kleine_karekiet", 2),
    ("Grote Karekiet", "Acrocephalus arundinaceus", "grote_karekiet", 3),
    ("Rietzanger", "Acrocephalus schoenobaenus", "rietzanger", 2),
    ("Bosrietzanger", "Acrocephalus palustris", "bosrietzanger", 2),
    ("Snor", "Locustella luscinioides", "snor", 3),
    ("Sprinkhaanzanger", "Locustella naevia", "sprinkhaanzanger", 3),
    ("Cetti's Zanger", "Cettia cetti", "cettis_zanger", 2),
    ("Baardman", "Panurus biarmicus", "baardman", 2),

    # Vliegenvangers
    ("Bonte Vliegenvanger", "Ficedula hypoleuca", "bonte_vliegenvanger", 2),
    ("Grauwe Vliegenvanger", "Muscicapa striata", "grauwe_vliegenvanger", 2),

    # Duiven
    ("Houtduif", "Columba palumbus", "houtduif", 1),
    ("Holenduif", "Columba oenas", "holenduif", 2),
    ("Stadsduif", "Columba livia", "stadsduif", 1),
    ("Turkse Tortel", "Streptopelia decaocto", "turkse_tortel", 1),
    ("Zomertortel", "Streptopelia turtur", "zomertortel", 3),

    # Kraaiachtigen
    ("Ekster", "Pica pica", "ekster", 1),
    ("Zwarte Kraai", "Corvus corone", "zwarte_kraai", 1),
    ("Bonte Kraai", "Corvus cornix", "bonte_kraai", 3),
    ("Roek", "Corvus frugilegus", "roek", 1),
    ("Kauw", "Coloeus monedula", "kauw", 1),
    ("Raaf", "Corvus corax", "raaf", 2),
    ("Gaai", "Garrulus glandarius", "gaai", 1),
    ("Notenkraker", "Nucifraga caryocatactes", "notenkraker", 3),

    # Spechten
    ("Grote Bonte Specht", "Dendrocopos major", "grote_bonte_specht", 1),
    ("Kleine Bonte Specht", "Dryobates minor", "kleine_bonte_specht", 2),
    ("Middelste Bonte Specht", "Dendrocoptes medius", "middelste_bonte_specht", 3),
    ("Groene Specht", "Picus viridis", "groene_specht", 1),
    ("Zwarte Specht", "Dryocopus martius", "zwarte_specht", 2),
    ("Draaihals", "Jynx torquilla", "draaihals", 3),

    # Boomkruipers en Boomklevers
    ("Boomkruiper", "Certhia brachydactyla", "boomkruiper", 1),
    ("Taigaboomkruiper", "Certhia familiaris", "taigaboomkruiper", 3),
    ("Boomklever", "Sitta europaea", "boomklever", 1),

    # Spreeuwen en Wielewaal
    ("Spreeuw", "Sturnus vulgaris", "spreeuw", 1),
    ("Wielewaal", "Oriolus oriolus", "wielewaal", 2),

    # Leeuweriken
    ("Veldleeuwerik", "Alauda arvensis", "veldleeuwerik", 2),
    ("Boomleeuwerik", "Lullula arborea", "boomleeuwerik", 2),
    ("Kuifleeuwerik", "Galerida cristata", "kuifleeuwerik", 3),

    # Zwaluwen
    ("Boerenzwaluw", "Hirundo rustica", "boerenzwaluw", 1),
    ("Huiszwaluw", "Delichon urbicum", "huiszwaluw", 1),
    ("Oeverzwaluw", "Riparia riparia", "oeverzwaluw", 2),

    # Gierzwaluwen
    ("Gierzwaluw", "Apus apus", "gierzwaluw", 1),

    # ============================================================
    # PRIORITEIT 2: WATERVOGELS EN ROOFVOGELS
    # ============================================================

    # Ganzen
    ("Grauwe Gans", "Anser anser", "grauwe_gans", 1),
    ("Kolgans", "Anser albifrons", "kolgans", 1),
    ("Brandgans", "Branta leucopsis", "brandgans", 2),
    ("Canadese Gans", "Branta canadensis", "canadese_gans", 2),
    ("Nijlgans", "Alopochen aegyptiaca", "nijlgans", 1),
    ("Kleine Rietgans", "Anser brachyrhynchus", "kleine_rietgans", 2),
    ("Taigarietgans", "Anser fabalis", "taigarietgans", 2),
    ("Toendrarietgans", "Anser serrirostris", "toendrarietgans", 2),
    ("Rotgans", "Branta bernicla", "rotgans", 2),
    ("Indische Gans", "Anser indicus", "indische_gans", 3),

    # Zwanen
    ("Knobbelzwaan", "Cygnus olor", "knobbelzwaan", 1),
    ("Wilde Zwaan", "Cygnus cygnus", "wilde_zwaan", 2),
    ("Kleine Zwaan", "Cygnus bewickii", "kleine_zwaan", 2),

    # Eenden
    ("Wilde Eend", "Anas platyrhynchos", "wilde_eend", 1),
    ("Krakeend", "Mareca strepera", "krakeend", 2),
    ("Smient", "Mareca penelope", "smient", 2),
    ("Slobeend", "Spatula clypeata", "slobeend", 2),
    ("Wintertaling", "Anas crecca", "wintertaling", 2),
    ("Zomertaling", "Spatula querquedula", "zomertaling", 3),
    ("Pijlstaart", "Anas acuta", "pijlstaart", 2),
    ("Tafeleend", "Aythya ferina", "tafeleend", 2),
    ("Kuifeend", "Aythya fuligula", "kuifeend", 1),
    ("Topper", "Aythya marila", "topper", 3),
    ("Eider", "Somateria mollissima", "eider", 2),
    ("Brilduiker", "Bucephala clangula", "brilduiker", 2),
    ("Grote Zaagbek", "Mergus merganser", "grote_zaagbek", 2),
    ("Middelste Zaagbek", "Mergus serrator", "middelste_zaagbek", 2),
    ("Nonnetje", "Mergellus albellus", "nonnetje", 2),
    ("Bergeend", "Tadorna tadorna", "bergeend", 2),
    ("Casarca", "Tadorna ferruginea", "casarca", 3),
    ("Mandarijneend", "Aix galericulata", "mandarijneend", 2),
    ("Carolina-eend", "Aix sponsa", "carolina_eend", 3),

    # Futen
    ("Fuut", "Podiceps cristatus", "fuut", 1),
    ("Dodaars", "Tachybaptus ruficollis", "dodaars", 2),
    ("Geoorde Fuut", "Podiceps nigricollis", "geoorde_fuut", 3),
    ("Roodhalsfuut", "Podiceps grisegena", "roodhalsfuut", 3),
    ("Kuifduiker", "Podiceps auritus", "kuifduiker", 3),

    # Reigers
    ("Blauwe Reiger", "Ardea cinerea", "blauwe_reiger", 1),
    ("Grote Zilverreiger", "Ardea alba", "grote_zilverreiger", 2),
    ("Kleine Zilverreiger", "Egretta garzetta", "kleine_zilverreiger", 2),
    ("Purperreiger", "Ardea purpurea", "purperreiger", 3),
    ("Roerdomp", "Botaurus stellaris", "roerdomp", 2),
    ("Woudaap", "Ixobrychus minutus", "woudaap", 3),
    ("Kwak", "Nycticorax nycticorax", "kwak", 3),
    ("Koereiger", "Bubulcus ibis", "koereiger", 3),

    # Ooievaars
    ("Ooievaar", "Ciconia ciconia", "ooievaar", 2),
    ("Zwarte Ooievaar", "Ciconia nigra", "zwarte_ooievaar", 3),

    # Lepelaar
    ("Lepelaar", "Platalea leucorodia", "lepelaar", 2),

    # Aalscholvers
    ("Aalscholver", "Phalacrocorax carbo", "aalscholver", 1),

    # Rallen
    ("Meerkoet", "Fulica atra", "meerkoet", 1),
    ("Waterhoen", "Gallinula chloropus", "waterhoen", 1),
    ("Waterral", "Rallus aquaticus", "waterral", 2),
    ("Porseleinhoen", "Porzana porzana", "porseleinhoen", 3),
    ("Klein Waterhoen", "Zapornia parva", "klein_waterhoen", 3),
    ("Kraanvogel", "Grus grus", "kraanvogel", 2),

    # Steltlopers
    ("Scholekster", "Haematopus ostralegus", "scholekster", 1),
    ("Kluut", "Recurvirostra avosetta", "kluut", 2),
    ("Kievit", "Vanellus vanellus", "kievit", 1),
    ("Goudplevier", "Pluvialis apricaria", "goudplevier", 2),
    ("Zilverplevier", "Pluvialis squatarola", "zilverplevier", 2),
    ("Bontbekplevier", "Charadrius hiaticula", "bontbekplevier", 2),
    ("Kleine Plevier", "Charadrius dubius", "kleine_plevier", 2),
    ("Strandplevier", "Charadrius alexandrinus", "strandplevier", 3),
    ("Grutto", "Limosa limosa", "grutto", 2),
    ("Rosse Grutto", "Limosa lapponica", "rosse_grutto", 2),
    ("Wulp", "Numenius arquata", "wulp", 2),
    ("Regenwulp", "Numenius phaeopus", "regenwulp", 2),
    ("Tureluur", "Tringa totanus", "tureluur", 2),
    ("Groenpootruiter", "Tringa nebularia", "groenpootruiter", 2),
    ("Zwarte Ruiter", "Tringa erythropus", "zwarte_ruiter", 2),
    ("Witgat", "Tringa ochropus", "witgat", 2),
    ("Bosruiter", "Tringa glareola", "bosruiter", 2),
    ("Oeverloper", "Actitis hypoleucos", "oeverloper", 2),
    ("Kemphaan", "Calidris pugnax", "kemphaan", 2),
    ("Drieteenstrandloper", "Calidris alba", "drieteenstrandloper", 2),
    ("Bonte Strandloper", "Calidris alpina", "bonte_strandloper", 2),
    ("Kanoetstrandloper", "Calidris canutus", "kanoetstrandloper", 2),
    ("Krombekstrandloper", "Calidris ferruginea", "krombekstrandloper", 3),
    ("Kleine Strandloper", "Calidris minuta", "kleine_strandloper", 2),
    ("Temminck's Strandloper", "Calidris temminckii", "temmincks_strandloper", 3),
    ("Watersnip", "Gallinago gallinago", "watersnip", 2),
    ("Houtsnip", "Scolopax rusticola", "houtsnip", 2),
    ("Bokje", "Lymnocryptes minimus", "bokje", 3),
    ("Steenloper", "Arenaria interpres", "steenloper", 2),

    # Meeuwen
    ("Kokmeeuw", "Chroicocephalus ridibundus", "kokmeeuw", 1),
    ("Stormmeeuw", "Larus canus", "stormmeeuw", 2),
    ("Zilvermeeuw", "Larus argentatus", "zilvermeeuw", 1),
    ("Kleine Mantelmeeuw", "Larus fuscus", "kleine_mantelmeeuw", 2),
    ("Grote Mantelmeeuw", "Larus marinus", "grote_mantelmeeuw", 2),
    ("Pontische Meeuw", "Larus cachinnans", "pontische_meeuw", 3),
    ("Dwergmeeuw", "Hydrocoloeus minutus", "dwergmeeuw", 3),
    ("Drieteenmeeuw", "Rissa tridactyla", "drieteenmeeuw", 3),

    # Sterns
    ("Visdief", "Sterna hirundo", "visdief", 2),
    ("Grote Stern", "Thalasseus sandvicensis", "grote_stern", 2),
    ("Dwergstern", "Sternula albifrons", "dwergstern", 3),
    ("Noordse Stern", "Sterna paradisaea", "noordse_stern", 3),
    ("Zwarte Stern", "Chlidonias niger", "zwarte_stern", 2),

    # Roofvogels
    ("Buizerd", "Buteo buteo", "buizerd", 1),
    ("Ruigpootbuizerd", "Buteo lagopus", "ruigpootbuizerd", 2),
    ("Sperwer", "Accipiter nisus", "sperwer", 1),
    ("Havik", "Accipiter gentilis", "havik", 2),
    ("Bruine Kiekendief", "Circus aeruginosus", "bruine_kiekendief", 2),
    ("Blauwe Kiekendief", "Circus cyaneus", "blauwe_kiekendief", 2),
    ("Grauwe Kiekendief", "Circus pygargus", "grauwe_kiekendief", 3),
    ("Rode Wouw", "Milvus milvus", "rode_wouw", 2),
    ("Zwarte Wouw", "Milvus migrans", "zwarte_wouw", 3),
    ("Zeearend", "Haliaeetus albicilla", "zeearend", 2),
    ("Visarend", "Pandion haliaetus", "visarend", 2),
    ("Wespendief", "Pernis apivorus", "wespendief", 2),

    # Valken
    ("Torenvalk", "Falco tinnunculus", "torenvalk", 1),
    ("Boomvalk", "Falco subbuteo", "boomvalk", 2),
    ("Slechtvalk", "Falco peregrinus", "slechtvalk", 2),
    ("Smelleken", "Falco columbarius", "smelleken", 2),

    # Uilen
    ("Bosuil", "Strix aluco", "bosuil", 1),
    ("Ransuil", "Asio otus", "ransuil", 2),
    ("Velduil", "Asio flammeus", "velduil", 2),
    ("Kerkuil", "Tyto alba", "kerkuil", 2),
    ("Steenuil", "Athene noctua", "steenuil", 2),
    ("Oehoe", "Bubo bubo", "oehoe", 3),

    # Hoenders
    ("Fazant", "Phasianus colchicus", "fazant", 1),
    ("Patrijs", "Perdix perdix", "patrijs", 2),
    ("Kwartel", "Coturnix coturnix", "kwartel", 2),

    # IJsvogel en Bijeneter
    ("IJsvogel", "Alcedo atthis", "ijsvogel", 2),
    ("Bijeneter", "Merops apiaster", "bijeneter", 3),
    ("Hop", "Upupa epops", "hop", 3),

    # Koekoek
    ("Koekoek", "Cuculus canorus", "koekoek", 2),

    # Tapuit en verwanten
    ("Tapuit", "Oenanthe oenanthe", "tapuit", 2),
    ("Paapje", "Saxicola rubetra", "paapje", 2),
    ("Roodborsttapuit", "Saxicola rubicola", "roodborsttapuit", 2),

    # ============================================================
    # PRIORITEIT 3: MINDER ALGEMENE SOORTEN
    # ============================================================

    # Zeldzamere zangers
    ("Orpheusspotvogel", "Hippolais polyglotta", "orpheusspotvogel", 3),
    ("Graszanger", "Cisticola juncidis", "graszanger", 3),

    # Zeldzamere steltlopers
    ("Steltkluut", "Himantopus himantopus", "steltkluut", 3),
    ("Morinelplevier", "Charadrius morinellus", "morinelplevier", 3),

    # Klauwieren
    ("Klapekster", "Lanius excubitor", "klapekster", 2),
    ("Grauwe Klauwier", "Lanius collurio", "grauwe_klauwier", 2),
    ("Roodkopklauwier", "Lanius senator", "roodkopklauwier", 3),

    # Pelikaan
    ("Roze Pelikaan", "Pelecanus onocrotalus", "roze_pelikaan", 3),

    # Aalscholvers
    ("Kuifaalscholver", "Gulosus aristotelis", "kuifaalscholver", 3),

    # Ibissen
    ("Zwarte Ibis", "Plegadis falcinellus", "zwarte_ibis", 3),
    ("Heilige Ibis", "Threskiornis aethiopicus", "heilige_ibis", 3),

    # Flamingo
    ("Flamingo", "Phoenicopterus roseus", "flamingo", 3),
]


def get_species_by_priority(priority: int = None) -> list:
    """Haal soorten op, optioneel gefilterd op prioriteit."""
    if priority is None:
        return DUTCH_BIRD_SPECIES
    return [s for s in DUTCH_BIRD_SPECIES if s[3] == priority]


def get_all_species_for_training() -> list:
    """Haal alle soorten op in training volgorde (prioriteit 1 eerst)."""
    sorted_species = sorted(DUTCH_BIRD_SPECIES, key=lambda x: (x[3], x[0]))
    return [(s[0], s[1], s[2]) for s in sorted_species]


def count_species() -> dict:
    """Tel soorten per prioriteit."""
    counts = {1: 0, 2: 0, 3: 0}
    for s in DUTCH_BIRD_SPECIES:
        counts[s[3]] += 1
    counts['total'] = len(DUTCH_BIRD_SPECIES)
    return counts


if __name__ == "__main__":
    counts = count_species()
    print(f"\nNederlandse Vogelsoorten voor Training")
    print("=" * 50)
    print(f"Prioriteit 1 (zeer algemeen): {counts[1]} soorten")
    print(f"Prioriteit 2 (regelmatig):    {counts[2]} soorten")
    print(f"Prioriteit 3 (minder algemeen): {counts[3]} soorten")
    print("-" * 50)
    print(f"TOTAAL: {counts['total']} soorten")
