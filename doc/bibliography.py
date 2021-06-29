# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"

import warnings
from pybtex.richtext import Text, Tag, HRef
from pybtex.style.formatting import BaseStyle


class IEEEStyle(BaseStyle):
    def format_article(self, param):
        entry = param["entry"]
        
        try:
            authors = []
            for author in entry.persons["author"]:
                text = ""
                if author.first_names is not None:
                    text += " ".join([s[0] + "." for s in author.first_names])
                    text += " "
                if author.middle_names is not None:
                    text += " ".join([s[0] + "." for s in author.middle_names])
                    text += " "
                if author.prelast_names is not None:
                    text += " ".join([s for s in author.prelast_names])
                    text += " "
                text += " ".join([s for s in author.last_names])
                authors.append(Text(text + ", "))
            
            title = ""
            in_protected = False
            for char in entry.fields["title"]:
                if char == "{":
                    in_protected = True
                elif char == "}":
                    in_protected = False
                else:
                    if in_protected:
                        title += char
                    else:
                        # Capitalize title in unprotected areas
                        if len(title) == 0:
                            title += char.upper()
                        else:
                            title += char.lower()
            title = Text('"', title, '," ')
            
            journal = Text(Tag("em", entry.fields["journal"]), ", ")
            
            if "volume" in entry.fields:
                volume = Text("vol. ", entry.fields["volume"], ", ")
            else:
                volume = Text()
            
            if "pages" in entry.fields:
                pages = Text("pp. ", entry.fields["pages"], ", ")
            else:
                pages = Text()
            
            date = entry.fields["year"]
            if "month" in entry.fields:
                date = entry.fields["month"] + " " + date
            date = Text(date, ". ")
            
            if "doi" in entry.fields: 
                doi = Text("doi: ", HRef(
                    "https://doi.org/" + entry.fields["doi"],
                    entry.fields["doi"]
                ))
            else:
                doi = Text()
            
            return Text(*authors, title, journal, volume, pages, date, doi)
        
        except:
            warnings.warn(f"Invalid BibTeX entry '{entry.key}'")
            return Text(entry.key)