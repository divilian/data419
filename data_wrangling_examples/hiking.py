
# I downloaded this file from data.world:
# https://data.world/city-of-ny/i8f4-bu5r/workspace/file?filename=directory-of-hiking-trails-1.json

import json
import pandas as pd

pd.set_option("display.width",200)
pd.set_option("display.max.columns",20)

# Since it's in JSON format, I tried Pandas read_json() function, which seemed
# to work okay. ("ht" stands for "hiking trails".)
ht = pd.read_json("directory-of-hiking-trails-1.json")

# After looking at the columns, I decided only to keep these. (lat and lon were
# all null; bummer.)
ht = ht[['Name','Park_Name','Length','Difficulty','Accessible',
    'Limited_Access']]

# The "Length" column has entries like:
#   0.8 miles
#   1.0 mile
#   Various
#   5.3 miles along four trails
#   None
# What I want, of course, is something numeric I can work with. So I'll get rid
# of the words "mile" or "miles" anywhere they occur; change anything else to
# zero, and convert the whole column to a double dtype.
length_col = ht.Length
length_col = length_col.str.replace(" miles","")
length_col = length_col.str.replace(" mile","")
length_col = pd.to_numeric(length_col, errors='coerce').fillna(0)
ht['Length'] = length_col

# Running ht.Difficulty.unique() exposed that there were some data entry errors
# (one row has "Easy " with a space on the end, instead of "Easy" as four
# characters), so I manually fixed these. 
diff_col = ht.Difficulty.copy()
diff_col[diff_col == "Easy "] = "Easy"
diff_col[diff_col == ""] = None
ht.Difficulty = diff_col

print(ht)
