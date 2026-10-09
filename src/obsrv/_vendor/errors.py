class CaptureFailedError(RuntimeError):
    @classmethod
    def analysis_failed(cls, diagnostic):
        return cls(diagnostic)
