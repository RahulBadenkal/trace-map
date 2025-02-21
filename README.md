# Given a shape
# Total perimeter
# A map
-> Return the route on the given map

To setup
>> pip install -r requirements.txt

To run
>> python main.py



* Create a graph representation of cubbon (nodes + edges). Around 80 nodes and 750 edges
* Find all paths in the graph
   - that starts from a given starting point
   - ends on the same starting point
   - must be at least 5000m long
   - cant visit the same edge more than twice
- For every path found above calculate `Discrete Fréchet Distance` with the provided shape
- Which ever path has the smallest distance is the closest route