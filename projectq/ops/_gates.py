#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
Contains definitions of standard gates such as
* Hadamard (H)
* Pauli-X (X / NOT)
* Pauli-Z (Z)
* T and its inverse (T / Tdagger)
* Swap gate (Swap)
* Phase gate (Ph)
* Rotation-Z (Rz)
* Phase-shift (R)
* Measurement (Measure)

and meta gates, i.e.,
* Allocate / Deallocate qubits
* Flush gate (end of circuit)
"""

import math
import cmath
import numpy as np

from projectq.ops import get_inverse
from ._basics import (BasicGate,
                      SelfInverseGate,
                      BasicRotationGate,
                      ClassicalInstructionGate,
                      FastForwardingGate)


class HGate(SelfInverseGate):
        """ Hadamard gate class """
        def __init__(self):
                super(HGate, self).__init__()
                self._matrix = 1. / cmath.sqrt(2.) * np.matrix([[1, 1], [1, -1]])

        def __str__(self):
                return "H"

H = HGate()


class IdentityGate(SelfInverseGate):
        """ Identity gate class """
        def __init__(self):
                super(IdentityGate, self).__init__()
                self._matrix =  np.matrix([[1, 0], [0, 1]])

        def __str__(self):
                return "Id"


I = Identity = IdentityGate()


class XGate(SelfInverseGate):
        """ Pauli-X gate class """
        def __init__(self):
                super(XGate, self).__init__()
                self._matrix = np.matrix([[0, 1], [1, 0]])

        def __str__(self):
                return "X"

X = NOT = XGate()


class YGate(SelfInverseGate):
        """ Pauli-Y gate class """
        def __init__(self):
                super(YGate, self).__init__()
                self._matrix = np.matrix([[0, -1j], [1j, 0]])

        def __str__(self):
                return "Y"

Y = YGate()


class ZGate(SelfInverseGate):
        """ Pauli-Z gate class """
        def __init__(self):
                super(ZGate, self).__init__()
                self._matrix = np.matrix([[1, 0], [0, -1]])

        def __str__(self):
                return "Z"


Z = ZGate()


class SGate(BasicGate):
        """ S gate class """
        def __init__(self):
                super(SGate, self).__init__()
                self._matrix = np.matrix([[1, 0], [0, 1j]])

        def __str__(self):
                return "S"

S = SGate()
Sdag = Sdagger = get_inverse(S)


class TGate(BasicGate):
        """ T gate class """
        def __init__(self):
                super(TGate, self).__init__()
                self._matrix = np.matrix([[1, 0], [0, cmath.exp(1j * cmath.pi / 4)]])

        def __str__(self):
                return "T"

T = TGate()
Tdag = Tdagger = get_inverse(T)


class SwapGate(SelfInverseGate):
        """ Swap gate class (swaps 2 qubits) """
        def __init__(self):
                super(SwapGate, self).__init__()
                self.interchangeable_qubit_indices = [[0, 1]]
                self._matrix = np.matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

        def __str__(self):
                return "Swap"

Swap = SwapGate()


class EntangleGate(BasicGate):
        """
        Entangle gate (Hadamard on first qubit, followed by CNOTs applied to all
        other qubits).
        """
        # TODO: This is nonsense for the python simulator.
        def __init__(self):
                super(EntangleGate, self).__init__()

        def __str__(self):
                return "Entangle"

Entangle = EntangleGate()


class Ph(BasicRotationGate):
        """ Phase gate (global phase) """
        def __init__(self, angle):
                super(Ph, self).__init__(angle)
                self._matrix = np.matrix([[cmath.exp(1j * self._angle), 0],
                                  [0, cmath.exp(1j * self._angle)]])


class Rx(BasicRotationGate):
        """ RotationX gate class """
        def __init__(self, angle):
                super(Rx, self).__init__(angle)
                self._matrix = np.matrix([[math.cos(0.5 * self._angle),
                                   -1j * math.sin(0.5 * self._angle)],
                                        [-1j * math.sin(0.5 * self._angle),
                                         math.cos(0.5 * self._angle)]])


class Ry(BasicRotationGate):
        """ RotationY gate class """
        def __init__(self, angle):
                super(Ry, self).__init__(angle)
                self._matrix = np.matrix([[math.cos(0.5 * self._angle),
                                   -math.sin(0.5 * self._angle)],
                                        [math.sin(0.5 * self._angle),
                                         math.cos(0.5 * self._angle)]])


class Rz(BasicRotationGate):
        """ RotationZ gate class """
        def __init__(self, angle):
                super(Rz, self).__init__(angle)
                self._matrix = np.matrix([[cmath.exp(-.5 * 1j * self._angle), 0],
                                  [0, cmath.exp(.5 * 1j * self._angle)]])


class R(BasicRotationGate):
        """ Phase-shift gate (equivalent to Rz up to a global phase) """
        def __init__(self, angle):
                super(R, self).__init__(angle)
                self._matrix = np.matrix([[1, 0], [0, cmath.exp(1j * self._angle)]])


class FlushGate(FastForwardingGate):
        """
        Flush gate (denotes the end of the circuit).

        Note:
                All compiler engines (cengines) which cache/buffer gates are obligated to
                flush and send all gates to the next compiler engine (followed by the
                flush command).
        
        Note:
                This gate is sent when calling
                
                .. code-block:: python
                
                        eng.flush()
                
                on the MainEngine `eng`.
        """

        def __str__(self):
                return ""


class MeasureGate(FastForwardingGate):
        """ Measurement gate class """
        def __str__(self):
                return "Measure"

Measure = MeasureGate()


class AllocateQubitGate(ClassicalInstructionGate):
        """ Qubit allocation gate class """
        def __str__(self):
                return "Allocate"

        def get_inverse(self):
                return DeallocateQubitGate()

Allocate = AllocateQubitGate()


class DeallocateQubitGate(FastForwardingGate):
        """ Qubit deallocation gate class """
        def __str__(self):
                return "Deallocate"

        def get_inverse(self):
                return Allocate

Deallocate = DeallocateQubitGate()


class AllocateDirtyQubitGate(ClassicalInstructionGate):
        """ Dirty qubit allocation gate class """
        def __str__(self):
                return "AllocateDirty"

        def get_inverse(self):
                return Deallocate

AllocateDirty = AllocateDirtyQubitGate()
