import validators
import tldextract


def is_valid_url(url: str) -> bool:
    """
    Check if URL is properly formatted
    """
    return validators.url(url)


def extract_domain(url: str):
    """
    Extract domain information
    """
    extracted = tldextract.extract(url)

    return {
        "subdomain": extracted.subdomain,
        "domain": extracted.domain,
        "suffix": extracted.suffix,
        "full_domain": f"{extracted.domain}.{extracted.suffix}"
    }