# Exceptions API

## Error Hierarchy

```
ApexRAGError (APEX_000)
├── ConfigurationError (APEX_001)
│   └── InvalidProviderError (APEX_002)
├── DocumentNotFoundError (APEX_100)
├── DocumentExistsError (APEX_101)
├── IngestionError (APEX_102)
├── QueryError (APEX_200)
├── ProviderError (APEX_201)
├── VerificationError (APEX_202)
├── StorageError (APEX_300)
│   └── DatabaseConnectionError (APEX_301)
├── AuthenticationError (APEX_400)
├── RateLimitError (APEX_401)
└── FileValidationError (APEX_402)
```

## Base Exception

::: apex_rag.exceptions.ApexRAGError
    handler: python
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
      docstring_style: google
      members:
        - code
        - message
        - hint
        - status_code
        - to_dict

## Configuration Errors

::: apex_rag.exceptions.ConfigurationError
    handler: python
    options:
      show_root_heading: true
      heading_level: 3

::: apex_rag.exceptions.InvalidProviderError
    handler: python
    options:
      show_root_heading: true
      heading_level: 3

## Document Errors

::: apex_rag.exceptions.DocumentNotFoundError
    handler: python
    options:
      show_root_heading: true
      heading_level: 3

::: apex_rag.exceptions.DocumentExistsError
    handler: python
    options:
      show_root_heading: true
      heading_level: 3

::: apex_rag.exceptions.IngestionError
    handler: python
    options:
      show_root_heading: true
      heading_level: 3

## Query Errors

::: apex_rag.exceptions.QueryError
    handler: python
    options:
      show_root_heading: true
      heading_level: 3

::: apex_rag.exceptions.ProviderError
    handler: python
    options:
      show_root_heading: true
      heading_level: 3

::: apex_rag.exceptions.VerificationError
    handler: python
    options:
      show_root_heading: true
      heading_level: 3

## Storage Errors

::: apex_rag.exceptions.StorageError
    handler: python
    options:
      show_root_heading: true
      heading_level: 3

::: apex_rag.exceptions.DatabaseConnectionError
    handler: python
    options:
      show_root_heading: true
      heading_level: 3

## API Errors

::: apex_rag.exceptions.AuthenticationError
    handler: python
    options:
      show_root_heading: true
      heading_level: 3

::: apex_rag.exceptions.RateLimitError
    handler: python
    options:
      show_root_heading: true
      heading_level: 3

::: apex_rag.exceptions.FileValidationError
    handler: python
    options:
      show_root_heading: true
      heading_level: 3
