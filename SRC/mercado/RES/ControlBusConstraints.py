#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pyomo.environ as pyo


class VoltageControlConstraints:

    @staticmethod
    def add_constraints(
        model,
        T,
        V_PU,
        V_ESP
    ):

        for item in V_ESP:

            if item == 0:
                continue

            barra, vesp = item

            for t in range(T):

                setattr(
                    model,
                    f"voltage_control_{t}_{barra}",
                    pyo.Constraint(
                        expr=V_PU[t, barra] == vesp
                    )
                )