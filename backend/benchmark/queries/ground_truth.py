"""
Ground Truth Parser — structured extraction from the benchmark bible.

Parses the synthetic geotechnical corpus ground truth document into
structured NeighborhoodProfile and ProjectRecord objects for query synthesis.
"""
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NeighborhoodProfile:
    """Geological profile for a neighborhood."""
    name: str
    area_number: int
    location: str = ""
    stratigraphy: str = ""
    spt_values: str = ""
    groundwater: str = ""
    foundations: str = ""
    challenges: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "area_number": self.area_number,
            "location": self.location,
            "stratigraphy": self.stratigraphy,
            "spt_values": self.spt_values,
            "groundwater": self.groundwater,
            "foundations": self.foundations,
            "challenges": self.challenges,
        }


@dataclass
class ProjectRecord:
    """A single project from the ground truth seed."""
    id: int
    project_num: str  # e.g. "BG11-005"
    project_name: str
    neighborhood: str
    year: int
    client: str
    project_type: str
    scope: str
    key_finding: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "project_num": self.project_num,
            "project_name": self.project_name,
            "neighborhood": self.neighborhood,
            "year": self.year,
            "client": self.client,
            "project_type": self.project_type,
            "scope": self.scope,
            "key_finding": self.key_finding,
        }


@dataclass
class GroundTruthCorpus:
    """Complete parsed ground truth."""
    neighborhoods: List[NeighborhoodProfile] = field(default_factory=list)
    projects: List[ProjectRecord] = field(default_factory=list)
    regional_geology: str = ""
    standards_info: str = ""
    raw_text: str = ""

    def get_neighborhood(self, name: str) -> Optional[NeighborhoodProfile]:
        """Find neighborhood by name (case-insensitive partial match)."""
        name_lower = name.lower()
        for n in self.neighborhoods:
            if name_lower in n.name.lower():
                return n
        return None

    def get_projects_by_neighborhood(self, neighborhood: str) -> List[ProjectRecord]:
        """Get all projects in a neighborhood."""
        name_lower = neighborhood.lower()
        return [p for p in self.projects if name_lower in p.neighborhood.lower()]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "neighborhoods": [n.to_dict() for n in self.neighborhoods],
            "projects": [p.to_dict() for p in self.projects],
            "n_neighborhoods": len(self.neighborhoods),
            "n_projects": len(self.projects),
        }


class GroundTruthParser:
    """Parse the ground truth bible into structured data."""

    # The 8 neighborhoods and their area numbers
    NEIGHBORHOODS = {
        1: "Westmount",
        2: "Aspen Ridge",
        3: "Ironwood Flats",
        4: "Valleyview Heights",
        5: "Railtown",
        6: "Crystal Shores",
        7: "Harvest Hills",
        8: "City Centre",
    }

    def parse(self, bible_path: str) -> GroundTruthCorpus:
        """Parse the ground truth file into structured data."""
        path = Path(bible_path)
        if not path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {bible_path}")

        text = path.read_text(encoding="utf-8", errors="ignore")

        corpus = GroundTruthCorpus(raw_text=text)

        # Parse sections
        corpus.regional_geology = self._extract_section(text, "1. Regional Geology", "2. Neighborhood")
        corpus.standards_info = self._extract_section(text, "3. Local Standards", "4. Project History")
        corpus.neighborhoods = self._parse_neighborhoods(text)
        corpus.projects = self._parse_projects(text)

        logger.info(
            f"Parsed ground truth: {len(corpus.neighborhoods)} neighborhoods, "
            f"{len(corpus.projects)} projects"
        )
        return corpus

    def _extract_section(self, text: str, start_marker: str, end_marker: str) -> str:
        """Extract text between two section markers."""
        start = text.find(start_marker)
        if start == -1:
            return ""
        end = text.find(end_marker, start + len(start_marker))
        if end == -1:
            return text[start:]
        return text[start:end].strip()

    def _parse_neighborhoods(self, text: str) -> List[NeighborhoodProfile]:
        """Parse all 8 neighborhood profiles from section 2."""
        neighborhoods = []

        for area_num, name in self.NEIGHBORHOODS.items():
            # Find the area section
            pattern = f"Area {area_num}"
            start = text.find(pattern)
            if start == -1:
                continue

            # Find the end (next Area or section 3)
            next_area = text.find(f"Area {area_num + 1}", start + 10)
            section_3 = text.find("3. Local Standards", start + 10)

            if next_area != -1:
                end = next_area
            elif section_3 != -1:
                end = section_3
            else:
                end = start + 2000  # Fallback

            section_text = text[start:end]

            profile = NeighborhoodProfile(
                name=name,
                area_number=area_num,
            )

            # Extract fields by looking for keywords
            profile.location = self._extract_field(section_text, "Location")
            profile.stratigraphy = self._extract_stratigraphy(section_text)
            profile.spt_values = self._extract_field(section_text, "SPT")
            profile.groundwater = self._extract_field(section_text, "Groundwater")
            profile.foundations = self._extract_field(section_text, "Foundations")
            profile.challenges = self._extract_field(section_text, "Challenges")

            neighborhoods.append(profile)

        return neighborhoods

    def _extract_field(self, text: str, keyword: str) -> str:
        """Extract the value for a field identified by keyword."""
        # Find the keyword
        idx = text.find(keyword)
        if idx == -1:
            return ""

        # Get text from keyword to next known field or double newline
        rest = text[idx:]
        # Find next field boundary
        next_fields = ["SPT", "Groundwater", "Foundations", "Challenges", "Area ", "Stratigraphy"]
        end = len(rest)
        for nf in next_fields:
            if nf == keyword:
                continue
            pos = rest.find(nf, len(keyword) + 1)
            if 0 < pos < end:
                end = pos

        value = rest[:end].strip()
        # Clean up: remove the keyword prefix
        if value.startswith(keyword):
            value = value[len(keyword):].strip()
        # Remove leading punctuation
        value = value.lstrip(":. ")
        return value

    def _extract_stratigraphy(self, text: str) -> str:
        """Extract stratigraphy data (depth ranges and soil types)."""
        idx = text.find("Stratigraphy")
        if idx == -1:
            # Try finding depth patterns directly
            idx = text.find("0.0 -")
            if idx == -1:
                return ""

        rest = text[idx:]
        end = len(rest)
        for marker in ["SPT", "Groundwater"]:
            pos = rest.find(marker)
            if 0 < pos < end:
                end = pos

        value = rest[:end].strip()
        if value.startswith("Stratigraphy"):
            value = value[len("Stratigraphy"):].strip()
        return value

    def _parse_projects(self, text: str) -> List[ProjectRecord]:
        """Parse all 30 projects from section 4."""
        projects = []

        # Find the project history section
        section_start = text.find("4. Project History")
        if section_start == -1:
            section_start = text.find("Project History Seed")
        if section_start == -1:
            return []

        project_text = text[section_start:]

        # Parse each project line
        # Format: ID\tProject#\tProjectName\tNeighborhood\tYear\tClient\tType\tScope\tKeyFinding
        # The data appears as numbered entries separated by the ID column
        lines = project_text.split('\n')

        # Look for lines that start with a number (project ID)
        current_project = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try to match a project entry starting with its numeric ID
            match = re.match(r'^(\d+)\s*(BG\d{2}-\w+)\s*(.+)', line)
            if match:
                if current_project:
                    projects.append(current_project)

                proj_id = int(match.group(1))
                proj_num = match.group(2)
                rest = match.group(3)

                # Parse the remaining fields - they're tab or multi-space separated
                parts = re.split(r'\t+', rest) if '\t' in rest else re.split(r'\s{2,}', rest)
                parts = [p.strip() for p in parts if p.strip()]

                current_project = ProjectRecord(
                    id=proj_id,
                    project_num=proj_num,
                    project_name=parts[0] if len(parts) > 0 else "",
                    neighborhood=parts[1] if len(parts) > 1 else "",
                    year=int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0,
                    client=parts[3] if len(parts) > 3 else "",
                    project_type=parts[4] if len(parts) > 4 else "",
                    scope=parts[5] if len(parts) > 5 else "",
                    key_finding=parts[6] if len(parts) > 6 else "",
                )

        if current_project:
            projects.append(current_project)

        logger.info(f"Parsed {len(projects)} projects from ground truth")
        return projects
