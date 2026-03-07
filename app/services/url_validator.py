from app.utils.url_utils import is_valid_url, extract_domain


class URLValidator:

    @staticmethod
    def analyze(url: str):

        if not is_valid_url(url):
            return {
                "valid": False,
                "reason": "Invalid URL format"
            }

        domain_info = extract_domain(url)

        return {
            "valid": True,
            "domain_info": domain_info
        }