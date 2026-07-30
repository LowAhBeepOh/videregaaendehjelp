# Anbefalinger for Videregående Hjelp+ (Pluss)

Dette dokumentet inneholder anbefalinger for hva slags innhold, funksjoner og forbedringer
som kan legges til for brukere med Videregående Hjelp+ (Pluss) – spesifikt for
Møre og Romsdal-elever.

---

## 1. Nyheter spesifikke for Møre og Romsdal

### Struktur i `News/news-data.json`
Hver nyhet har nå et `"plus": true/false`-felt. For å opprette en Pluss-nyhet,
legg til `"plus": true` og eventuelt `"county": "Møre og Romsdal"`:

```json
{
  "name": "Skolestart 2026 – viktige datoer for MR",
  "desc": "Fylkeskommunen har offentliggjort datoer for skolestart og inntak i Møre og Romsdal.",
  "thumb": "https://eksempel.com/bilde.jpg",
  "date": "2026-07-15",
  "link": "News/mr-skolestart.html",
  "plus": true,
  "county": "Møre og Romsdal"
}
```

### Forslag til Pluss-nyheter:
- **Skolestartinfo**: Datoer, frister og praktisk info for MR-elever
- **Inntak og poenggrenser**: Oppdaterte poenggrenser for MR-skoler
- **Eksamensplan**: Lokale eksamensdatoer og -steder i MR
- **Lærlingplasser**: Oversikt over lærlingplasser i MR-regionen
- **Arrangementer**: Karrieredager, åpne dager og andre skolearrangementer i MR
- **Politikk**: Nyheter fra MR fylkeskommune som påvirker elever
- **Samferdsel**: Skoleskyss-ruter, endringer i buss/tog-tilbud for MR-elever

---

## 2. Hurtiginfo/data-seksjon

En ny seksjon på `index.html` (eller en egen Pluss-side) som viser:

- **Skolestartnedtelling**: Antall dager til skolestart
- **VIGO-lenker**: Direktelenker til VIGO-søknadssystemet for MR
- **Udir-oppdateringer**: Siste nytt fra Udir om eksamen og regler
- **Vilbli.no-lenker**: Programoversikt spesifikt for MR
- **MR fylke "skolestart"-side**: Direktelenke til fylkets offisielle skolestartside
- **Værmelding**: For skoleområder i MR (krever gratis API som Open-Meteo)

### Eksempel-implementasjon (JS-komponent):
```javascript
// Hurtiginfo for Pluss-brukere – viser MR-spesifikk data
function renderPlusQuickInfo() {
  if (!window.VHplus || !VHplus.isActive()) return;
  // Hent data fra en JSON-fil eller statisk innhold
  // Vis på index.html eller egen side
}
```

---

## 3. Nye temaer og stilinnstillinger

Pluss-brukere kan få tilgang til flere visuelle tilpasninger:

### Temaforslag:
- **MR-fylkestema**: Farger inspirert av Møre og Romsdals fylkesvåpen
  - Blåtoner fra kysten og grønt fra fjordene
  - Alternativ logo med MR-profil
- **Eksamenstema**: Rolige, fokuserte farger for lesing
  - Dempede toner, høy kontrast, minimalt med farger
- **Skjermvennlig**: Optimalisert for skole-PC-er og projektorer
  - Større skrift, høyere kontrast, forenklet layout

### Nye innstillinger:
- Egen `data-font-size`-innstilling for å skalere tekst
- Eget `data-layout`-attributt (kompakt/romslig) for guide-visning
- "LESA"-modus (Lettere å SE Alt) – høy kontrast + stor skrift
- Mulighet for å skjule/seksjoner i feeden på index.html

---

## 4. Feed-kontroll

Pluss-brukere skal ha "full kontroll over hva de vil se på Feeden".
Dette kan implementeres som:

- **Filter-knapper** på index.html: Vis/skjul guider, verktøy, nyheter, fun facts
- **Lagrede layout-preferanser**: Hvilke seksjoner som vises og i hvilken rekkefølge
- **Skjulte guider**: Brukeren kan "skjule" guider de ikke trenger
- **Prioriterte emner**: Velg fag/områder som skal vises først

### Lagring:
```javascript
// Feed-innstillinger lagres i VHplus settings
VHplus.set({
  settings: {
    feed: {
      showGuides: true,
      showTools: true,
      showNews: true,
      showFunFacts: true,
      hiddenGuides: [],
      preferredSubjects: [],
    }
  }
});
```

---

## 5. Beta-funksjoner

Tidlig tilgang til nye funksjoner:

- **Eksamenssimulator beta**: Nye eksamenssett og funksjoner før alle andre
- **Studieplanlegger**: Lag personlige timeplaner og studieplaner
- **Karriereveileder**: AI-basert karriereanbefaling (hvis implementert)
- **Ressursbank**: Eksklusive sammendrag, flashcards og øvingssett
- **Samarbeidsrom**: Del notater og tips med andre MR-elever

---

## 6. Data-synkronisering i skyen

Pluss-data (innstillinger, preferanser) bør synkroniseres til Supabase
`profiles`-tabellen. Dette kan lagres i `sync_data`-feltet under en `plus`-nøkkel:

```javascript
// I plus.js eller account.js
var plusData = VHplus.get();
await VHaccount.updateProfile({
  sync_data: {
    ...existingSyncData,
    plus: plusData,
  },
});
```

---

## 7. Tekniske anbefalinger

- Opprett en egen JSON-fil: `News/plus-news-data.json` for Pluss-spesifikke nyheter,
  eller hold alt i `news-data.json` med `"plus": true` (anbefalt for enkelhet)
- For hurtiginfo, opprett `Assets/js/plus-quick-info.js` som en egen modul
- For feed-kontroll, utvid `index.html` med et kontrollpanel på toppen av bento-grinden
- Merk: `plus.js` modulen er allerede inkludert på alle hovedpages via `<script>`-tag

---

## 8. Eksempel: Oppsett av ny Pluss-funksjon

1. Legg til nytt innhold i riktig JSON-fil med `"plus": true`
2. Hvis ny JS-funksjonalitet trengs, lag en ny modul i `Assets/js/` og inkluder den
3. Hvis ny styling trengs, legg til i `Assets/css/plus.css` eller lag egen CSS-fil
4. Oppdater filter-logikken i `index.html` eller `news.html` for å sjekke `VHplus.isActive()`
