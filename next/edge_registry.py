import pandas as pd

class EdgeRegistry:

    def __init__(self):
        self.edges = pd.DataFrame()

    # ------------------------------------------
    # Add results from EdgeMiner
    # ------------------------------------------
    def register(self, edge_results):

        df_new = pd.DataFrame(edge_results)

        if self.edges.empty:
            self.edges = df_new
        else:
            self.edges = pd.concat([self.edges, df_new], ignore_index=True)

        return self.edges

    # ------------------------------------------
    # Ranking
    # ------------------------------------------
    def rank(self, by="global_sharpe", ascending=False):

        if self.edges.empty:
            return self.edges

        return self.edges.sort_values(by=by, ascending=ascending)

    # ------------------------------------------
    # Filter by threshold
    # ------------------------------------------
    def filter(self,
               min_global_sharpe=None,
               min_regime_sharpe=None):

        df = self.edges.copy()

        if min_global_sharpe is not None:
            df = df[df["global_sharpe"] >= min_global_sharpe]

        if min_regime_sharpe is not None:
            df = df[df["best_regime_sharpe"] >= min_regime_sharpe]

        return df

    # ------------------------------------------
    # Get top N
    # ------------------------------------------
    def top_n(self, n=5, by="global_sharpe"):

        return self.rank(by=by).head(n)

    # ------------------------------------------
    # Summary stats
    # ------------------------------------------
    def summary(self):

        if self.edges.empty:
            return "No edges registered."

        return {
            "total_edges": len(self.edges),
            "mean_global_sharpe": self.edges["global_sharpe"].mean(),
            "max_global_sharpe": self.edges["global_sharpe"].max(),
            "mean_regime_sharpe": self.edges["best_regime_sharpe"].mean()
        }

    # ------------------------------------------
    # Save to CSV (for QC export)
    # ------------------------------------------
    def save(self, path="edge_registry.csv"):

        self.edges.to_csv(path, index=False)
