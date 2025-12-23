# BirdNET-Pi Vocalization Classifier - Community Pitch

## Wat is het?

Een uitbreiding voor BirdNET-Pi die automatisch herkent of een vogel **zingt**, **roept** of **alarmeert**. Dit geeft extra context bij elke detectie.

## Waarom is dit waardevol?

| Vocalisatie | Betekenis | Voorbeeld |
|-------------|-----------|-----------|
| **Zang** | Territorium markeren, partner zoeken | Merel zingt 's ochtends |
| **Roep** | Contact houden, locatie aangeven | Koolmees roept naar groepsgenoten |
| **Alarm** | Gevaar! Predator in de buurt | Roodborst waarschuwt voor kat |

### Praktische toepassingen:
- **Gedragsonderzoek**: Wanneer zingen vogels? Hoe reageert de populatie op verstoringen?
- **Predator detectie**: Alarmroepen kunnen wijzen op katten, sperwers, of andere predatoren
- **Seizoenspatronen**: Zangactiviteit volgen door het jaar heen
- **Data verrijking**: Meer informatie uit dezelfde opnames halen

## Hoe werkt het?

```
BirdNET detecteert "Merel"
    ↓
Vocalization Classifier analyseert audio
    ↓
Resultaat: "Merel - Zang (93%)"
```

### Technisch:
1. **Training**: CNN model getraind op spectrogrammen van Xeno-canto opnames
2. **Per soort**: Elk model is specifiek getraind voor één vogelsoort
3. **Lichtgewicht**: ~2MB per model, draait op Raspberry Pi
4. **Nauwkeurig**: 85-95% accuracy voor veel voorkomende soorten

## Huidige status

- **196 getrainde modellen** (Nederlandse vogelsoorten)
- **76.000+ detecties** verwerkt in productie
- **Draait 27+ dagen** stabiel op Raspberry Pi 5
- **Open source**: MIT licentie

## Training pipeline

```
Xeno-canto audio → Spectrogrammen → CNN Training → Model per soort
```

De training kan lokaal (Docker) of via Google Colab (gratis GPU).

## Integratie opties

### Optie A: Standalone addon
- Aparte service naast BirdNET-Pi
- Leest `birds.db`, voegt kolom toe
- Geen wijzigingen aan BirdNET-Pi core

### Optie B: Native integratie
- Pull request naar BirdNET-Pi
- Vocalisatie zichtbaar in web interface
- Naadloze gebruikerservaring

### Optie C: Fork met uitbreidingen
- BirdNET-Pi Plus met extra features
- Onderhouden door community

## Wat is er nodig?

### Voor standalone:
- Raspberry Pi 4/5 met BirdNET-Pi
- ~400MB voor modellen
- Python 3.9+ met PyTorch

### Voor integratie in BirdNET-Pi:
- Review door Nachtzuster
- UI aanpassingen voor weergave
- Documentatie

## Demo

Bekijk het in actie:
- **Grafana dashboard**: Vocalisatie statistieken per dag/soort
- **LED display**: Real-time notificaties met vocalisatie type

## Contact

- **GitHub**: https://github.com/RonnyCHL/emsn-vocalization
- **Auteur**: Ronny Hullegie (EMSN - Ecologisch Monitoring Systeem Nijverdal)

## Vragen?

1. **Interesse in testen?** - Modellen beschikbaar voor download
2. **Wil je helpen trainen?** - Colab notebook beschikbaar
3. **Suggesties voor integratie?** - Open een issue

---

*Dit project is ontwikkeld als onderdeel van EMSN (Ecologisch Monitoring Systeem Nijverdal) en vrij beschikbaar voor de BirdNET-Pi community.*
