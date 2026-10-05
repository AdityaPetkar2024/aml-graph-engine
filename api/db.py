from neo4j import GraphDatabase

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

Q_WRITE = """
MERGE (a:Account {id: $src})
MERGE (b:Account {id: $dst})
MERGE (a)-[t:TX {tx_id: $tx_id}]->(b)
ON CREATE SET t.ts = $ts, t.amt_paid = $amt_paid, t.fmt = $fmt, t.label = $label
"""

# Micro-batch write. CREATE (not MERGE) for edges: MERGE on a hub account would scan all
# its edges per insert. The simulator's --reset removes test-period edges first, so a
# replay never duplicates them.
Q_WRITE_BATCH = """
UNWIND $rows AS r
MERGE (a:Account {id: r.src})
MERGE (b:Account {id: r.dst})
CREATE (a)-[:TX {tx_id: r.tx_id, ts: r.ts, amt_paid: r.amt_paid, fmt: r.fmt, label: r.label}]->(b)
"""

Q_HOP1 = """
MATCH (a:Account {id: $id})-[t:TX]-(b:Account)
WHERE t.ts >= $since AND a <> b
WITH t ORDER BY t.ts DESC LIMIT $limit
RETURN startNode(t).id AS src, endNode(t).id AS dst,
       t.amt_paid AS amount, t.ts AS ts, t.fmt AS fmt
"""

Q_HOP2 = """
UNWIND $ids AS nid
MATCH (b:Account {id: nid})
CALL (b) {
  MATCH (b)-[t:TX]-(c:Account)
  WHERE t.ts >= $since AND b <> c AND c.id <> $id
  RETURN t ORDER BY t.ts DESC LIMIT $per
}
RETURN startNode(t).id AS src, endNode(t).id AS dst,
       t.amt_paid AS amount, t.ts AS ts, t.fmt AS fmt
"""


class Graph:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def ping(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    def write_tx(self, tx: dict):
        with self.driver.session() as s:
            s.run(Q_WRITE, src=tx["src"], dst=tx["dst"], tx_id=tx["tx_id"],
                  ts=int(tx["ts"].timestamp()), amt_paid=tx["amt_paid"],
                  fmt=tx["fmt"], label=tx.get("label")).consume()

    def write_batch(self, txs: list[dict]):
        rows = [{"src": t["src"], "dst": t["dst"], "tx_id": t["tx_id"],
                 "ts": int(t["ts"].timestamp()), "amt_paid": t["amt_paid"],
                 "fmt": t["fmt"], "label": t.get("label")} for t in txs]
        with self.driver.session() as s:
            s.run(Q_WRITE_BATCH, rows=rows).consume()

    def subgraph(self, account: str, since: int, hop1: int = 40, per: int = 5):
        """Two-hop neighbourhood, capped so hub accounts can't blow up the result."""
        with self.driver.session() as s:
            e1 = s.run(Q_HOP1, id=account, since=since, limit=hop1).data()
            nbrs = list({e["dst"] if e["src"] == account else e["src"] for e in e1})
            e2 = s.run(Q_HOP2, ids=nbrs, id=account, since=since, per=per).data() if nbrs else []
        seen, edges = set(), []
        for e in e1 + e2:
            key = (e["src"], e["dst"], e["ts"], e["amount"])
            if key not in seen:
                seen.add(key)
                edges.append(e)
        return edges
