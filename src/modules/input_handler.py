class InputHandler:
    """
    Handles processing and validation of user inputs for keyword research
    """
    
    def __init__(self):
        self.min_keyword_length = 2
        self.max_keywords = 10
    
    def process_input(self, data):
        """
        Process and validate user input
        
        Args:
            data (dict): User input data containing seed keywords and optional parameters
            
        Returns:
            dict: Processed input data
        """
        # Extract and validate primary seed keyword
        seed_keyword = data.get('seed_keyword', '').strip()
        
        if not seed_keyword:
            raise ValueError("Seed keyword is required")
        
        if len(seed_keyword) < self.min_keyword_length:
            raise ValueError(f"Seed keyword must be at least {self.min_keyword_length} characters")
        
        # Extract optional parameters
        secondary_keywords = data.get('secondary_keywords', [])
        industry = data.get('industry', '')
        competitor_urls = data.get('competitor_urls', [])
        
        # Validate secondary keywords
        if isinstance(secondary_keywords, str):
            secondary_keywords = [kw.strip() for kw in secondary_keywords.split(',') if kw.strip()]
        
        if len(secondary_keywords) > self.max_keywords:
            raise ValueError(f"Maximum {self.max_keywords} secondary keywords allowed")
        
        # Validate competitor URLs
        if isinstance(competitor_urls, str):
            competitor_urls = [url.strip() for url in competitor_urls.split(',') if url.strip()]
        
        # Create expanded keyword list
        keywords = [seed_keyword] + secondary_keywords
        
        # Normalize and deduplicate keywords
        keywords = list(set([kw.lower() for kw in keywords]))
        
        # Generate keyword variations
        variations = self._generate_variations(keywords)
        
        return {
            'seed_keyword': seed_keyword,
            'secondary_keywords': secondary_keywords,
            'keywords': keywords,
            'variations': variations,
            'industry': industry,
            'competitor_urls': competitor_urls
        }
    
    def _generate_variations(self, keywords):
        """
        Generate variations of the keywords for expanded research
        
        Args:
            keywords (list): List of keywords
            
        Returns:
            list: Expanded list of keyword variations
        """
        variations = []
        
        for keyword in keywords:
            # Add basic variations like questions
            variations.append(f"what is {keyword}")
            variations.append(f"how to {keyword}")
            variations.append(f"best {keyword}")
            variations.append(f"{keyword} guide")
            
            # Add commercial variations
            variations.append(f"{keyword} price")
            variations.append(f"buy {keyword}")
            variations.append(f"{keyword} review")
            
        # Return unique variations
        return list(set(variations))
