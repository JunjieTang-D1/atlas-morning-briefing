"""Frontend component stubs for Clinical Dashboard (Python test helpers)."""


def render_patient_list(patients: list) -> str:
    """Render patient list as HTML table for testing."""
    if not patients:
        return "<div class=\"empty\">No patients found</div>"

    rows = []
    for p in patients:
        rows.append(
            f"<tr><td>{p.get('name', '')}</td>"
            f"<td>{p.get('mrn', '')}</td>"
            f"<td>{p.get('dob', '')}</td></tr>"
        )
    return f"<table><thead><tr><th>Name</th><th>MRN</th><th>DOB</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"


def render_search_bar() -> str:
    """Render search bar component."""
    return '<input type="text" id="patient-search" placeholder="Search patients..." />'
