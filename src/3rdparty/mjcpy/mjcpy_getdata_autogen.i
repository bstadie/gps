    out["nstack"] = m_data->nstack;
    out["nbuffer"] = m_data->nbuffer;
    out["pstack"] = m_data->pstack;
    out["maxstackuse"] = m_data->maxstackuse;
    out["ne"] = m_data->ne;
    out["nc"] = m_data->nc;
    out["nfri"] = m_data->nfri;
    out["nlim"] = m_data->nlim;
    out["ncon"] = m_data->ncon;
    out["nflc"] = m_data->nflc;
    out["nstat"] = m_data->nstat;
    out["time"] = m_data->time;
    out["com"] = toNdarray1<mjtNum>(m_data->com, 3);
    out["energy"] = toNdarray1<mjtNum>(m_data->energy, 2);
    out["fwdinv"] = toNdarray1<mjtNum>(m_data->fwdinv, 4);
    out["userdata"] = toNdarray2<mjtNum>(m_data->userdata, m_model->nuserdata, 1);
    out["qpos"] = toNdarray2<mjtNum>(m_data->qpos, m_model->nq, 1);
    out["qvel"] = toNdarray2<mjtNum>(m_data->qvel, m_model->nv, 1);
    out["qvel_next"] = toNdarray2<mjtNum>(m_data->qvel_next, m_model->nv, 1);
    out["act"] = toNdarray2<mjtNum>(m_data->act, m_model->na, 1);
    out["act_next"] = toNdarray2<mjtNum>(m_data->act_next, m_model->na, 1);
    out["ctrl"] = toNdarray2<mjtNum>(m_data->ctrl, m_model->nu, 1);
    out["qfrc_applied"] = toNdarray2<mjtNum>(m_data->qfrc_applied, m_model->nv, 1);
    out["xfrc_applied"] = toNdarray2<mjtNum>(m_data->xfrc_applied, m_model->nbody, 6);
    out["qfrc_bias"] = toNdarray2<mjtNum>(m_data->qfrc_bias, m_model->nv, 1);
    out["qfrc_passive"] = toNdarray2<mjtNum>(m_data->qfrc_passive, m_model->nv, 1);
    out["qfrc_actuation"] = toNdarray2<mjtNum>(m_data->qfrc_actuation, m_model->nv, 1);
    out["qfrc_impulse"] = toNdarray2<mjtNum>(m_data->qfrc_impulse, m_model->nv, 1);
    out["qfrc_constraint"] = toNdarray2<mjtNum>(m_data->qfrc_constraint, m_model->nv, 1);
    out["xpos"] = toNdarray2<mjtNum>(m_data->xpos, m_model->nbody, 3);
    out["xquat"] = toNdarray2<mjtNum>(m_data->xquat, m_model->nbody, 4);
    out["xmat"] = toNdarray2<mjtNum>(m_data->xmat, m_model->nbody, 9);
    out["xipos"] = toNdarray2<mjtNum>(m_data->xipos, m_model->nbody, 3);
    out["ximat"] = toNdarray2<mjtNum>(m_data->ximat, m_model->nbody, 9);
    out["xanchor"] = toNdarray2<mjtNum>(m_data->xanchor, m_model->njnt, 3);
    out["xaxis"] = toNdarray2<mjtNum>(m_data->xaxis, m_model->njnt, 3);
    out["geom_xpos"] = toNdarray2<mjtNum>(m_data->geom_xpos, m_model->ngeom, 3);
    out["geom_xmat"] = toNdarray2<mjtNum>(m_data->geom_xmat, m_model->ngeom, 9);
    out["site_xpos"] = toNdarray2<mjtNum>(m_data->site_xpos, m_model->nsite, 3);
    out["site_xmat"] = toNdarray2<mjtNum>(m_data->site_xmat, m_model->nsite, 9);
    out["cdof"] = toNdarray2<mjtNum>(m_data->cdof, m_model->nv, 6);
    out["cinert"] = toNdarray2<mjtNum>(m_data->cinert, m_model->nbody, 10);
    out["ten_wrapadr"] = toNdarray2<int>(m_data->ten_wrapadr, m_model->ntendon, 1);
    out["ten_wrapnum"] = toNdarray2<int>(m_data->ten_wrapnum, m_model->ntendon, 1);
    out["ten_length"] = toNdarray2<mjtNum>(m_data->ten_length, m_model->ntendon, 1);
    out["ten_moment"] = toNdarray2<mjtNum>(m_data->ten_moment, m_model->ntendon, m_model->nv);
    out["wrap_obj"] = toNdarray2<int>(m_data->wrap_obj, m_model->nwrap*2, 1);
    out["wrap_xpos"] = toNdarray2<mjtNum>(m_data->wrap_xpos, m_model->nwrap*2, 3);
    out["actuator_length"] = toNdarray2<mjtNum>(m_data->actuator_length, m_model->nu, 1);
    out["actuator_moment"] = toNdarray2<mjtNum>(m_data->actuator_moment, m_model->nu, m_model->nv);
    out["actuator_force"] = toNdarray2<mjtNum>(m_data->actuator_force, m_model->nu, 1);
    out["cvel"] = toNdarray2<mjtNum>(m_data->cvel, m_model->nbody, 6);
    out["cacc"] = toNdarray2<mjtNum>(m_data->cacc, m_model->nbody, 6);
    out["cfrc_int"] = toNdarray2<mjtNum>(m_data->cfrc_int, m_model->nbody, 6);
    out["cfrc_ext"] = toNdarray2<mjtNum>(m_data->cfrc_ext, m_model->nbody, 6);
    out["qM"] = toNdarray2<mjtNum>(m_data->qM, m_model->nM, 1);
    out["qD"] = toNdarray2<mjtNum>(m_data->qD, m_model->nv, 1);
    out["qLD"] = toNdarray2<mjtNum>(m_data->qLD, m_model->nM, 1);
    out["qLDiagSqr"] = toNdarray2<mjtNum>(m_data->qLDiagSqr, m_model->nv, 1);
    out["eq_id"] = toNdarray2<int>(m_data->eq_id, m_model->nemax, 1);
    out["eq_err"] = toNdarray2<mjtNum>(m_data->eq_err, m_model->nemax, 1);
    out["eq_J"] = toNdarray2<mjtNum>(m_data->eq_J, m_model->nemax, m_model->nv);
    out["eq_vdes"] = toNdarray2<mjtNum>(m_data->eq_vdes, m_model->nemax, 1);
    out["eq_JMi"] = toNdarray2<mjtNum>(m_data->eq_JMi, m_model->nemax, m_model->nv);
    out["eq_Achol"] = toNdarray2<mjtNum>(m_data->eq_Achol, m_model->nemax, m_model->nemax);
    out["eq_R"] = toNdarray2<mjtNum>(m_data->eq_R, m_model->nemax, 1);
    out["fri_id"] = toNdarray2<int>(m_data->fri_id, m_model->nv, 1);
    out["lim_id"] = toNdarray2<int>(m_data->lim_id, m_model->nlmax, 1);
    out["con_id"] = toNdarray2<int>(m_data->con_id, m_model->ncmax, 1);
    out["lc_ind"] = toNdarray2<int>(m_data->lc_ind, m_model->nlmax+m_model->ncmax, 1);
    out["flc_signature"] = toNdarray2<int>(m_data->flc_signature, m_model->njmax, 1);
    out["lc_dist"] = toNdarray2<mjtNum>(m_data->lc_dist, m_model->nlmax+m_model->ncmax, 1);
    out["flc_J"] = toNdarray2<mjtNum>(m_data->flc_J, m_model->njmax, m_model->nv);
    out["flc_ek"] = toNdarray2<mjtNum>(m_data->flc_ek, m_model->njmax, 2);
    out["flc_A"] = toNdarray2<mjtNum>(m_data->flc_A, m_model->njmax, m_model->njmax);
    out["flc_R"] = toNdarray2<mjtNum>(m_data->flc_R, m_model->njmax, 1);
    out["flc_vdes"] = toNdarray2<mjtNum>(m_data->flc_vdes, m_model->njmax, 1);
    out["flc_b"] = toNdarray2<mjtNum>(m_data->flc_b, m_model->njmax, 1);
    out["flc_f"] = toNdarray2<mjtNum>(m_data->flc_f, m_model->njmax, 1);