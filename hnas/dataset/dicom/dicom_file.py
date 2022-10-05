SOPInstanceUID = str

# Abstract class.
class DICOMFile:
    @property
    def path(self) -> str:
        raise NotImplementedError("Child class must implement 'path'.")
