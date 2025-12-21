#!/usr/bin/env python3
"""
EMSN 2.0 Vocalisatie Classifier - Xeno-canto Collector
Downloads vogelgeluiden van Xeno-canto met vocalisatie type labels.

Ondersteunt:
- API v3 (vereist API key: https://xeno-canto.org/explore/api)
- Web scraping fallback (geen key nodig, maar langzamer)
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote, urlencode

import requests
from tqdm import tqdm

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Recording:
    """Xeno-canto opname metadata."""
    id: str
    species: str
    species_common: str
    vocalization_type: str  # song, call, alarm, etc.
    quality: str  # A, B, C, D, E
    length: str  # Duration string
    country: str
    recordist: str
    url: str
    download_url: str
    license: str

    @property
    def filename(self) -> str:
        """Genereer bestandsnaam voor download."""
        voc_type = self.normalize_vocalization_type(self.vocalization_type)
        species_clean = self.species.replace(' ', '_').lower()
        return f"XC{self.id}_{species_clean}_{voc_type}.mp3"

    @staticmethod
    def normalize_vocalization_type(voc_type: str) -> str:
        """Normaliseer vocalisatie type naar song/call/alarm."""
        voc_lower = voc_type.lower()

        # Song patterns
        if any(p in voc_lower for p in ['song', 'zang', 'singing', 'subsong', 'dawn song']):
            return 'song'

        # Alarm patterns
        if any(p in voc_lower for p in ['alarm', 'warning', 'alert', 'distress', 'scolding']):
            return 'alarm'

        # Call patterns (default voor niet-zang)
        if any(p in voc_lower for p in ['call', 'roep', 'contact', 'flight', 'begging']):
            return 'call'

        # Als onduidelijk, check of het zang-achtig is
        if any(p in voc_lower for p in ['melodious', 'phrase', 'motif']):
            return 'song'

        # Default naar call voor overige
        return 'call'


class XenoCantoClient:
    """
    Client voor Xeno-canto.

    Gebruikt API v3 als api_key beschikbaar is, anders web scraping.
    """

    API_V3_URL = "https://xeno-canto.org/api/3/recordings"
    SEARCH_URL = "https://xeno-canto.org/explore"

    def __init__(
        self,
        download_dir: str = "data/raw/xeno-canto",
        api_key: Optional[str] = None
    ):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.environ.get('XENO_CANTO_API_KEY')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'EMSN-Vocalization-Classifier/1.0 (research project)'
        })
        self.request_delay = 2.0  # Respecteer rate limiting
        self._last_request = 0

    def _rate_limit(self):
        """Wacht indien nodig voor rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request = time.time()

    def search(
        self,
        species: str,
        vocalization_type: Optional[str] = None,
        quality: Optional[list[str]] = None,
        country: Optional[str] = None,
        max_results: int = 100
    ) -> list[Recording]:
        """
        Zoek opnames op Xeno-canto.
        """
        if self.api_key:
            return self._search_api(species, vocalization_type, quality, country, max_results)
        else:
            return self._search_scrape(species, vocalization_type, quality, country, max_results)

    def _search_api(
        self,
        species: str,
        vocalization_type: Optional[str] = None,
        quality: Optional[list[str]] = None,
        country: Optional[str] = None,
        max_results: int = 100
    ) -> list[Recording]:
        """Zoek via API v3 (vereist API key)."""
        # Bouw query voor API v3
        # Format: sp:"turdus merula" type:song q:A
        query_parts = [f'sp:"{species}"']

        if vocalization_type:
            query_parts.append(f'type:{vocalization_type}')

        if quality:
            for q in quality:
                query_parts.append(f'q:{q}')

        if country:
            query_parts.append(f'cnt:"{country}"')

        query = ' '.join(query_parts)
        logger.info(f"API v3 zoeken: {query}")

        self._rate_limit()

        try:
            response = self.session.get(
                self.API_V3_URL,
                params={'query': query, 'key': self.api_key, 'per_page': min(max_results, 500)},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error(f"API request mislukt: {e}")
            return []

        return self._parse_recordings(data.get('recordings', [])[:max_results])

    def _search_scrape(
        self,
        species: str,
        vocalization_type: Optional[str] = None,
        quality: Optional[list[str]] = None,
        country: Optional[str] = None,
        max_results: int = 100
    ) -> list[Recording]:
        """
        Zoek via web scraping (geen API key nodig).
        Haalt JSON data uit de explore pagina.
        """
        # Bouw zoekquery voor website
        # Format: Turdus+merula+type:song+q:A
        query_parts = [species.replace(' ', '+')]

        if vocalization_type:
            query_parts.append(f'type:{vocalization_type}')

        if quality:
            # Neem eerste kwaliteit voor simpelheid
            query_parts.append(f'q:{quality[0]}')

        if country:
            query_parts.append(f'cnt:{country}')

        query = '+'.join(query_parts)

        # Gebruik de JSON endpoint die de website intern gebruikt
        json_url = f"https://xeno-canto.org/api/internal/recordings?query={quote(query)}"

        logger.info(f"Web scraping zoeken: {species} (type={vocalization_type})")

        self._rate_limit()

        try:
            # Eerst de explore pagina bezoeken om cookies te krijgen
            self.session.get(f"https://xeno-canto.org/explore?query={quote(query)}", timeout=10)

            # Dan de interne API aanroepen
            response = self.session.get(json_url, timeout=30)

            if response.status_code == 200:
                try:
                    data = response.json()
                    recordings = data.get('recordings', [])
                    logger.info(f"Gevonden via interne API: {len(recordings)} opnames")
                    return self._parse_recordings(recordings[:max_results])
                except json.JSONDecodeError:
                    pass

            # Fallback: parse HTML pagina
            return self._parse_html_search(species, vocalization_type, quality, max_results)

        except requests.RequestException as e:
            logger.error(f"Web request mislukt: {e}")
            return []

    def _parse_html_search(
        self,
        species: str,
        vocalization_type: Optional[str] = None,
        quality: Optional[list[str]] = None,
        max_results: int = 100
    ) -> list[Recording]:
        """Parse zoekresultaten direct van HTML met paginatie."""
        # Bouw query - spaties als + en geen URL encoding van de rest
        query_parts = [species.replace(' ', '+')]
        if vocalization_type:
            query_parts.append(f'type:{vocalization_type}')
        if quality:
            query_parts.append(f'q:{quality[0]}')

        query = '+'.join(query_parts)

        all_ids = set()
        page = 1
        max_pages = (max_results // 30) + 2  # ~30 resultaten per pagina

        while len(all_ids) < max_results and page <= max_pages:
            # Geen quote() gebruiken - xeno-canto verwacht plain text
            url = f"https://xeno-canto.org/explore?query={query}&pg={page}"

            if page == 1:
                logger.info(f"HTML fallback: {url}")
            else:
                logger.debug(f"Pagina {page}: {url}")

            self._rate_limit()

            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                html = response.text
            except requests.RequestException as e:
                logger.error(f"HTML request mislukt pagina {page}: {e}")
                break

            # Zoek naar recording IDs in de HTML
            id_pattern = re.compile(r'xeno-canto\.org/(\d{5,8})"')
            page_ids = set(id_pattern.findall(html))

            if not page_ids:
                logger.debug(f"Geen resultaten meer op pagina {page}")
                break

            new_ids = page_ids - all_ids
            if not new_ids:
                logger.debug(f"Geen nieuwe IDs op pagina {page}")
                break

            all_ids.update(page_ids)
            logger.info(f"Pagina {page}: {len(page_ids)} IDs, totaal uniek: {len(all_ids)}")
            page += 1

        ids = list(all_ids)[:max_results]
        logger.info(f"Gevonden unieke IDs: {len(ids)}")

        # Haal details op per recording
        recordings = []
        for xc_id in tqdm(ids[:max_results], desc="Fetching details"):
            rec = self._get_recording_details(xc_id, species, vocalization_type)
            if rec:
                recordings.append(rec)

        return recordings

    def _get_recording_details(
        self,
        xc_id: str,
        species: str,
        vocalization_type: Optional[str] = None
    ) -> Optional[Recording]:
        """Haal details van een enkele opname."""
        self._rate_limit()

        try:
            url = f"https://xeno-canto.org/{xc_id}"
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            html = response.text

            # Download URL - gebruik de standaard download URL
            # De directe mp3 link vereist parsing maar /download werkt altijd
            download_url = f"https://xeno-canto.org/{xc_id}/download"

            # Type uit HTML - zoek in de metadata tabel
            type_match = re.search(r'Sound type[^<]*</td>\s*<td[^>]*>([^<]+)', html, re.IGNORECASE)
            if not type_match:
                type_match = re.search(r'>Type</td>\s*<td[^>]*>([^<]+)', html, re.IGNORECASE)
            voc_type = type_match.group(1).strip() if type_match else (vocalization_type or 'unknown')

            # Kwaliteit - zoek naar quality rating
            qual_match = re.search(r'class=["\']quality["\'][^>]*>([A-E])', html)
            if not qual_match:
                qual_match = re.search(r'Rating[^<]*</td>\s*<td[^>]*>([A-E])', html)
            quality = qual_match.group(1) if qual_match else 'C'

            # Lengte
            len_match = re.search(r'Length[^<]*</td>\s*<td[^>]*>([\d:]+)', html)
            if not len_match:
                len_match = re.search(r'duration.*?PT(\d+)M(\d+)S', html)
                if len_match:
                    length = f"{len_match.group(1)}:{len_match.group(2).zfill(2)}"
                else:
                    length = '0:00'
            else:
                length = len_match.group(1)

            # Land
            cnt_match = re.search(r'Country[^<]*</td>\s*<td[^>]*>([^<]+)', html)
            country = cnt_match.group(1).strip() if cnt_match else ''

            # Recordist
            rec_match = re.search(r'Recordist[^<]*</td>\s*<td[^>]*>(?:<[^>]+>)?([^<]+)', html)
            recordist = rec_match.group(1).strip() if rec_match else ''

            return Recording(
                id=xc_id,
                species=species,
                species_common='',
                vocalization_type=voc_type,
                quality=quality,
                length=length,
                country=country,
                recordist=recordist,
                url=url,
                download_url=download_url,
                license='CC'
            )

        except Exception as e:
            logger.warning(f"Kon details niet ophalen voor XC{xc_id}: {e}")
            return None

    def _parse_recordings(self, recordings_data: list[dict]) -> list[Recording]:
        """Parse recording data naar Recording objecten."""
        recordings = []

        for rec in recordings_data:
            try:
                recording = Recording(
                    id=str(rec.get('id', '')),
                    species=rec.get('sp', ''),
                    species_common=rec.get('en', ''),
                    vocalization_type=rec.get('type', 'unknown'),
                    quality=rec.get('q', 'E'),
                    length=rec.get('length', '0:00'),
                    country=rec.get('cnt', ''),
                    recordist=rec.get('rec', ''),
                    url=rec.get('url', ''),
                    download_url=rec.get('file', ''),
                    license=rec.get('lic', '')
                )
                recordings.append(recording)
            except Exception as e:
                logger.warning(f"Kon opname niet parsen: {e}")

        return recordings

    def download(
        self,
        recording: Recording,
        subdir: Optional[str] = None
    ) -> Optional[Path]:
        """Download een opname."""
        if subdir:
            target_dir = self.download_dir / subdir
        else:
            target_dir = self.download_dir

        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / recording.filename

        if target_path.exists():
            logger.debug(f"Al aanwezig: {target_path.name}")
            return target_path

        # Download URL opschonen
        download_url = recording.download_url
        if not download_url:
            download_url = f"https://xeno-canto.org/{recording.id}/download"
        elif download_url.startswith('//'):
            download_url = 'https:' + download_url
        elif not download_url.startswith('http'):
            download_url = 'https://' + download_url

        self._rate_limit()

        try:
            response = self.session.get(download_url, timeout=60, stream=True, allow_redirects=True)
            response.raise_for_status()

            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Verificeer dat het een geldig audio bestand is
            if target_path.stat().st_size < 1000:
                logger.warning(f"Bestand te klein, mogelijk geen audio: {target_path.name}")
                target_path.unlink()
                return None

            logger.debug(f"Gedownload: {target_path.name}")
            return target_path

        except requests.RequestException as e:
            logger.error(f"Download mislukt voor XC{recording.id}: {e}")
            if target_path.exists():
                target_path.unlink()
            return None

    def download_dataset(
        self,
        species: str,
        vocalization_types: list[str] = ['song', 'call', 'alarm'],
        quality: list[str] = ['A', 'B'],
        samples_per_type: int = 50
    ) -> dict[str, list[Path]]:
        """Download een complete dataset voor een soort."""
        dataset = {}

        for voc_type in vocalization_types:
            logger.info(f"\n{'='*50}")
            logger.info(f"Downloaden: {species} - {voc_type}")
            logger.info(f"{'='*50}")

            recordings = self.search(
                species=species,
                vocalization_type=voc_type,
                quality=quality,
                max_results=samples_per_type * 2
            )

            # Filter op genormaliseerd type
            filtered = [
                r for r in recordings
                if Recording.normalize_vocalization_type(r.vocalization_type) == voc_type
            ]

            logger.info(f"Na filtering: {len(filtered)} opnames voor '{voc_type}'")

            downloaded = []
            for rec in tqdm(filtered[:samples_per_type], desc=f"Downloading {voc_type}"):
                path = self.download(rec, subdir=voc_type)
                if path:
                    downloaded.append(path)

            dataset[voc_type] = downloaded
            logger.info(f"Gedownload: {len(downloaded)} bestanden voor '{voc_type}'")

        return dataset

    def save_metadata(self, recordings: list[Recording], filepath: str):
        """Sla metadata op als JSON."""
        data = []
        for rec in recordings:
            data.append({
                'id': rec.id,
                'species': rec.species,
                'species_common': rec.species_common,
                'vocalization_type': rec.vocalization_type,
                'vocalization_type_normalized': Recording.normalize_vocalization_type(rec.vocalization_type),
                'quality': rec.quality,
                'length': rec.length,
                'country': rec.country,
                'recordist': rec.recordist,
                'url': rec.url,
                'filename': rec.filename,
                'license': rec.license
            })

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata opgeslagen: {filepath}")


def main():
    """Download Merel dataset voor proof-of-concept."""
    import argparse

    parser = argparse.ArgumentParser(description='Download vogelgeluiden van Xeno-canto')
    parser.add_argument('--species', default='Turdus merula', help='Wetenschappelijke naam')
    parser.add_argument('--types', nargs='+', default=['song', 'call', 'alarm'],
                       help='Vocalisatie types')
    parser.add_argument('--quality', nargs='+', default=['A', 'B'],
                       help='Kwaliteiten (A=beste)')
    parser.add_argument('--samples', type=int, default=30,
                       help='Samples per type')
    parser.add_argument('--output', default='data/raw/xeno-canto',
                       help='Output directory')
    parser.add_argument('--api-key', default=None,
                       help='Xeno-canto API key (of set XENO_CANTO_API_KEY env var)')

    args = parser.parse_args()

    client = XenoCantoClient(download_dir=args.output, api_key=args.api_key)

    mode = "API v3" if client.api_key else "Web scraping"

    print(f"\n{'='*60}")
    print(f"EMSN 2.0 - Xeno-canto Dataset Downloader")
    print(f"{'='*60}")
    print(f"Modus: {mode}")
    print(f"Soort: {args.species}")
    print(f"Types: {', '.join(args.types)}")
    print(f"Kwaliteit: {', '.join(args.quality)}")
    print(f"Samples per type: {args.samples}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

    if not client.api_key:
        print("TIP: Vraag een gratis API key aan op https://xeno-canto.org/explore/api")
        print("     voor snellere en betrouwbaardere downloads.\n")

    dataset = client.download_dataset(
        species=args.species,
        vocalization_types=args.types,
        quality=args.quality,
        samples_per_type=args.samples
    )

    print(f"\n{'='*60}")
    print("SAMENVATTING")
    print(f"{'='*60}")

    total = 0
    for voc_type, files in dataset.items():
        print(f"  {voc_type}: {len(files)} bestanden")
        total += len(files)

    print(f"{'='*60}")
    print(f"  TOTAAL: {total} bestanden")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
