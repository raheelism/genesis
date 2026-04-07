from genesis.core.sdr import SDR
from genesis.perception.encoder import Encoder

THETA_VERBALIZE = 0.05   # minimum similarity to include a token candidate


class Verbalizer:
    """Converts a conclusion SDR back into a token sequence.
    Generation is concept-first: find tokens that best cover the SDR bits."""

    def verbalize(self, sdr: SDR, encoder: Encoder,
                  max_tokens: int = 10) -> str:
        if sdr.popcount() == 0:
            return "<unknown>"

        # Score all registered tokens by SDR similarity
        scored = [
            (token, sdr.similarity(tok_sdr))
            for token, tok_sdr in encoder._vocab_sdrs.items()
            if token not in ("<pad>", "<unk>", "<start>", "<end>")
        ]
        scored = [(tok, s) for tok, s in scored if s > THETA_VERBALIZE]
        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored:
            return "<unknown>"

        # Greedily pick tokens that cover the most uncovered SDR bits
        covered = SDR.zeros()
        selected = []
        for token, score in scored[:50]:   # search top-50 candidates
            tok_sdr = encoder.encode_token(token)
            new_coverage = tok_sdr.similarity(sdr) - covered.similarity(tok_sdr)
            if new_coverage > 0:
                selected.append(token)
                covered = covered.compose(tok_sdr)
                if len(selected) >= max_tokens:
                    break

        return " ".join(selected) if selected else "<unknown>"
