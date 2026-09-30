# RFC-0004: Local device execution and isolation

Status: current local execution boundary for device launchers in this source snapshot.

A hardware launcher runs on the machine holding the assigned card. It uses the
checked-in source tree, and `ASCRIPTOR_BOARDS` selects an ignored local JSON entry
with `"local": true`, a workspace, device visibility and the card's core count.
SSH destinations and connection fields are unsupported.

Generate inputs and an independent reference on that machine, execute the selected
artifact, and compare actual outputs. Keep the local workspace and output directory
isolated per task, hold the configured device lock and apply a bounded timeout.
Source emission, vendor compilation and card execution are separate evidence gates.
A model run cannot substitute for card acceptance.

See the [execution API](../api/execution.md#running-the-whole-unit-on-the-device-machine)
for the local invocation and the [agent runtime guide](../../../agent/en/runtime-and-maintenance.md#hardware-first)
for validation order. The source snapshot's `sources.json` selects the exact files.
