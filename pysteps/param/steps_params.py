from dataclasses import dataclass, field
import datetime

#                       95%         50%         5%
# nonzero_mean_db      6.883147    4.590082   2.815397
# nonzero_stdev_db     3.793680    2.489131   1.298552
# rain_fraction        0.447717    0.048889   0.008789
# beta_1              -0.452957   -1.681647  -2.726216
# beta_2              -2.322891   -3.251342  -4.009131
# corl_zero         1074.976508  188.058276  23.489147


@dataclass
class StepsParameters:
    metadata: dict

    # STEPS parameters with defaults for light rain
    nonzero_mean_db: float = 2.81
    nonzero_stdev_db: float = 1.3
    rain_fraction: float = 0
    beta_1: float = -2.05
    beta_2: float = -3.2
    corl_zero: float = 180

    # Auto-correlation lists
    lag_1: list[float] = field(default_factory=list)
    lag_2: list[float] = field(default_factory=list)

    # Required metadata keys
    _required_metadata_keys = {
        "domain",
        "product",
        "valid_time",
        "base_time",
        "ensemble",
    }

    def get(self, key: str, default=None):
        """Mimic dict.get(). Check metadata first, then top-level attributes."""
        if key in self.metadata:
            value = self.metadata.get(key)
            if (
                value is None
                and key in self._required_metadata_keys
                and default is None
            ):
                raise KeyError(f"Required metadata key '{key}' is missing or None.")
            return value if value is not None else default
        else:
            return getattr(self, key, default)

    def set_metadata(self, key: str, value):
        """Set a metadata key/value pair and validate if required."""
        self.metadata[key] = value
        if key in self._required_metadata_keys and value is None:
            raise ValueError(f"Required metadata key '{key}' cannot be None.")

    def validate(self):
        """Raise ValueError if any required field is missing or None."""
        for key in self._required_metadata_keys:
            if key not in self.metadata or self.metadata[key] is None:
                raise ValueError(f"Missing required metadata field: '{key}'")

    @classmethod
    def from_dict(cls, data: dict):
        """Create a StepsParameters object from a dictionary."""

        def ensure_utc(dt):
            if dt is None:
                return None
            if isinstance(dt, str):
                dt = datetime.datetime.fromisoformat(dt)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=datetime.timezone.utc)
            return dt.astimezone(datetime.timezone.utc)

        meta = data.get("metadata", {})
        if meta is not None:
            metadata = {
                "domain": meta.get("domain"),
                "product": meta.get("product"),
                "valid_time": ensure_utc(meta.get("valid_time")),
                "base_time": ensure_utc(meta.get("base_time")),
                "ensemble": meta.get("ensemble"),
            }
        else:
            metadata = {}

        return cls(
            metadata=metadata,
            nonzero_mean_db=data.get("nonzero_mean_db", 2.81),
            nonzero_stdev_db=data.get("nonzero_stdev_db", 1.3),
            rain_fraction=data.get("rain_fraction", 0),
            beta_1=data.get("beta_1", -2.05),
            beta_2=data.get("beta_2", -3.2),
            corl_zero=data.get("corl_zero", 180),
            lag_1=data.get("lag_1", []),
            lag_2=data.get("lag_2", []),
        )

    def to_dict(self):
        """Convert the object into a dictionary for MongoDB or JSON."""
        if self.metadata is not None:
            metadata = {
                "domain": self.metadata["domain"],
                "product": self.metadata["product"],
                "valid_time": self.metadata["valid_time"],
                "base_time": self.metadata["base_time"],
                "ensemble": self.metadata["ensemble"],
            }
        else:
            metadata = {}

        return {
            "metadata": metadata,
            "nonzero_mean_db": self.nonzero_mean_db,
            "nonzero_stdev_db": self.nonzero_stdev_db,
            "rain_fraction": self.rain_fraction,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "corl_zero": self.corl_zero,
            "lag_1": self.lag_1,
            "lag_2": self.lag_2,
        }
