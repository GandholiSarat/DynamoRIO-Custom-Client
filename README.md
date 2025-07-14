# DynamoRIO-Custom-Client

This project is a customized version of DynamoRIO's sample client `memtrace_x86`.

## Description

The primary modifications include:

- Custom buffer size configuration.
- Applying address offsets to instruction addresses in the trace output.

These changes are designed for specialized memory trace analysis where address normalization or differentiation is required.

## Trace Output Format

The trace output format remains unchanged from the original `memtrace_x86` client, except that the instruction address now reflects an offset.

## Advantage 
This client gives less overhead than the native client for the same trace format. 

