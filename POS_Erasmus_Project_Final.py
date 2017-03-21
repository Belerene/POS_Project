#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import timeit
import math
import random
import pickle
import numpy as np
os.environ["THEANO_FLAGS"] = "exception_verbosity=high"
import theano
import theano.tensor as T
from xml.etree import ElementTree as ET
theano.config.floatX = 'float32'

from sklearn.preprocessing import LabelBinarizer
from sklearn import naive_bayes


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

def getTagset():
    return ['Ncmsn','Ncmsg','Ncmsd','Ncmsan','Ncmsay','Ncmsl','Ncmsi','Ncmdn','Ncmdg','Ncmdd','Ncmda','Ncmdl','Ncmdi',
              'Ncmpn','Ncmpg','Ncmpd','Ncmpa','Ncmpl','Ncmpi','Ncfsn','Ncfsg','Ncfsd','Ncfsa','Ncfsl','Ncfsi','Ncfdn','Ncfdg',
              'Ncfdd','Ncfda','Ncfdl','Ncfdi','Ncfpn','Ncfpg','Ncfpd','Ncfpa','Ncfpl','Ncfpi','Ncnsn','Ncnsg','Ncnsd','Ncnsa',
              'Ncnsl','Ncnsi','Ncndn','Ncndg','Ncndd','Ncnda','Ncndl','Ncndi','Ncnpn','Ncnpg','Ncnpd','Ncnpa','Ncnpl','Ncnpi',
              'Npmsn','Npmsg','Npmsd','Npmsan','Npmsay','Npmsl','Npmsi','Npmdn','Npmdg','Npmdd','Npmda','Npmdl','Npmdi','Npmpn',
              'Npmpg','Npmpd','Npmpa','Npmpl','Npmpi','Npfsn','Npfsg','Npfsd','Npfsa','Npfsl','Npfsi','Npfdn','Npfdg','Npfdd',
              'Npfda','Npfdl','Npfdi','Npfpn','Npfpg','Npfpd','Npfpa','Npfpl','Npfpi','Npnsn','Npnsg','Npnsd','Npnsa','Npnsl',
              'Npnsi','Npnpn','Npnpg','Npnpd','Npnpa','Npnpl','Npnpi','Vmen','Vmeu','Vmep-sm','Vmep-sf','Vmep-sn','Vmep-pm',
              'Vmep-pf','Vmep-pn','Vmep-dm','Vmep-df','Vmep-dn','Vmer1s','Vmer1p','Vmer1d','Vmer2s','Vmer2p','Vmer2d','Vmer3s',
              'Vmer3p','Vmer3d','Vmem1p','Vmem1d','Vmem2s','Vmem2p','Vmem2d','Vmpn','Vmpu','Vmpp-sm','Vmpp-sf','Vmpp-sn','Vmpp-pm',
              'Vmpp-pf','Vmpp-pn','Vmpp-dm','Vmpp-df','Vmpp-dn','Vmpr1s','Vmpr1s-n','Vmpr1s-y','Vmpr1p','Vmpr1p-n','Vmpr1p-y',
              'Vmpr1d','Vmpr1d-n','Vmpr1d-y','Vmpr2s','Vmpr2s-n','Vmpr2s-y','Vmpr2p','Vmpr2p-n','Vmpr2p-y','Vmpr2d','Vmpr2d-n',
              'Vmpr2d-y','Vmpr3s','Vmpr3s-n','Vmpr3s-y','Vmpr3p','Vmpr3p-n','Vmpr3p-y','Vmpr3d','Vmpr3d-n','Vmpr3d-y','Vmpm1p',
              'Vmpm1d','Vmpm2s','Vmpm2p','Vmpm2d','Vmbn','Vmbu','Vmbp-sm','Vmbp-sf','Vmbp-sn','Vmbp-pm','Vmbp-pf','Vmbp-pn',
              'Vmbp-dm','Vmbp-df','Vmbp-dn','Vmbr1s','Vmbr1p','Vmbr1d','Vmbr1df','Vmbr2s','Vmbr2p','Vmbr2d','Vmbr3s','Vmbr3p',
              'Vmbr3d','Vmbf1s','Vmbf1p','Vmbf1d','Vmbf2s','Vmbf2p','Vmbf2d','Vmbf3s','Vmbf3p','Vmbf3d','Vmbm1p','Vmbm1d',
              'Vmbm2s','Vmbm2p','Vmbm2d','Va-n','Va-u','Va-p-sm','Va-p-sf','Va-p-sn','Va-p-pm','Va-p-pf','Va-p-pn','Va-p-dm',
              'Va-p-df','Va-p-dn','Va-r1s-n','Va-r1s-y','Va-r1p-n','Va-r1p-y','Va-r1d-n','Va-r1d-y','Va-r1dfn','Va-r2s-n',
              'Va-r2s-y','Va-r2p-n','Va-r2p-y','Va-r2d-n','Va-r2d-y','Va-r3s-n','Va-r3s-y','Va-r3p-n','Va-r3p-y','Va-r3d-n',
              'Va-r3d-y','Va-f1s-n','Va-f1s-y','Va-f1p-n','Va-f1p-y','Va-f1d','Va-f1d-n','Va-f1d-y','Va-f2s-n','Va-f2s-y',
              'Va-f2p-n','Va-f2d-n','Va-f3s-n','Va-f3s-y','Va-f3p-n','Va-f3p-y','Va-f3d-n','Va-c','Va-c---y','Va-m1p',
              'Va-m1d','Va-m2s','Va-m2p','Va-m2d','Agpmsnn','Agpmsny','Agpmsg','Agpmsd','Agpmsa','Agpmsan','Agpmsay','Agpmsl',
              'Agpmsi','Agpmdn','Agpmdg','Agpmdd','Agpmda','Agpmdl','Agpmdi','Agpmpn','Agpmpg','Agpmpd','Agpmpa','Agpmpl',
              'Agpmpi','Agpfsn','Agpfsg','Agpfsd','Agpfsa','Agpfsl','Agpfsi','Agpfdn','Agpfdg','Agpfdd','Agpfda','Agpfdl',
              'Agpfdi','Agpfpn','Agpfpg','Agpfpd','Agpfpa','Agpfpl','Agpfpi','Agpnsn','Agpnsg','Agpnsd','Agpnsa','Agpnsl',
              'Agpnsi','Agpndn','Agpndg','Agpndd','Agpnda','Agpndl','Agpndi','Agpnpn','Agpnpg','Agpnpd','Agpnpa','Agpnpl',
              'Agpnpi','Agcmsny','Agcmsg','Agcmsd','Agcmsa','Agcmsay','Agcmsl','Agcmsi','Agcmdn','Agcmdg','Agcmdd','Agcmda',
              'Agcmdl','Agcmdi','Agcmpn','Agcmpg','Agcmpd','Agcmpa','Agcmpl','Agcmpi','Agcfsn','Agcfsg','Agcfsd','Agcfsa',
              'Agcfsl','Agcfsi','Agcfdn','Agcfdg','Agcfdd','Agcfda','Agcfdl','Agcfdi','Agcfpn','Agcfpg','Agcfpd','Agcfpa',
              'Agcfpl','Agcfpi','Agcnsn','Agcnsg','Agcnsd','Agcnsa','Agcnsl','Agcnsi','Agcndn','Agcndg','Agcndd','Agcnda',
              'Agcndl','Agcndi','Agcnpn','Agcnpg','Agcnpd','Agcnpa','Agcnpl','Agcnpi','Agsmsny','Agsmsg','Agsmsd','Agsmsa',
              'Agsmsay','Agsmsl','Agsmsi','Agsmdn','Agsmdg','Agsmdd','Agsmda','Agsmdl','Agsmdi','Agsmpn','Agsmpg','Agsmpd',
              'Agsmpa','Agsmpl','Agsmpi','Agsfsn','Agsfsg','Agsfsd','Agsfsa','Agsfsl','Agsfsi','Agsfdn','Agsfdg','Agsfdd',
              'Agsfda','Agsfdl','Agsfdi','Agsfpn','Agsfpg','Agsfpd','Agsfpa','Agsfpl','Agsfpi','Agsnsn','Agsnsg','Agsnsd',
              'Agsnsa','Agsnsl','Agsnsi','Agsndn','Agsndg','Agsndd','Agsnda','Agsndl','Agsndi','Agsnpn','Agsnpg','Agsnpd',
              'Agsnpa','Agsnpl','Agsnpi','Aspmsnn','Aspmsg','Aspmsd','Aspmsa','Aspmsan','Aspmsl','Aspmsi','Aspmdn','Aspmdg',
              'Aspmdd','Aspmda','Aspmdl','Aspmdi','Aspmpn','Aspmpg','Aspmpd','Aspmpa','Aspmpl','Aspmpi','Aspfsn','Aspfsg',
              'Aspfsd','Aspfsa','Aspfsl','Aspfsi','Aspfdn','Aspfdg','Aspfdd','Aspfda','Aspfdl','Aspfdi','Aspfpn','Aspfpg',
              'Aspfpd','Aspfpa','Aspfpl','Aspfpi','Aspnsn','Aspnsg','Aspnsd','Aspnsa','Aspnsl','Aspnsi','Aspndn','Aspndg',
              'Aspndd','Aspnda','Aspndl','Aspndi','Aspnpn','Aspnpg','Aspnpd','Aspnpa','Aspnpl','Aspnpi','Appmsnn','Appmsny',
              'Appmsg','Appmsd','Appmsa','Appmsan','Appmsay','Appmsl','Appmsi','Appmdn','Appmdg','Appmdd','Appmda','Appmdl',
              'Appmdi','Appmpn','Appmpg','Appmpd','Appmpa','Appmpl','Appmpi','Appfsn','Appfsg','Appfsd','Appfsa','Appfsl',
              'Appfsi','Appfdn','Appfdg','Appfdd','Appfda','Appfdl','Appfdi','Appfpn','Appfpg','Appfpd','Appfpa','Appfpl',
              'Appfpi','Appnsn','Appnsg','Appnsd','Appnsa','Appnsl','Appnsi','Appndn','Appndg','Appndd','Appnda','Appndl',
              'Appndi','Appnpn','Appnpg','Appnpd','Appnpa','Appnpl','Appnpi','Rgp','Rgc','Rgs','Rr','Pp1-sn','Pp1-sg','Pp1-sg--y',
              'Pp1-sd','Pp1-sd--y','Pp1-sa','Pp1-sa--y','Pp1-sa--b','Pp1-sl','Pp1-si','Pp1-dg','Pp1-dd','Pp1-da','Pp1-dl',
              'Pp1-di','Pp1-pg','Pp1-pd','Pp1-pa','Pp1-pl','Pp1-pi','Pp1mdn','Pp1mpn','Pp1fdn','Pp1fpn','Pp1ndn','Pp1npn',
              'Pp2-sn','Pp2-sg','Pp2-sg--y','Pp2-sd','Pp2-sd--y','Pp2-sa','Pp2-sa--y','Pp2-sa--b','Pp2-sl','Pp2-si',
              'Pp2-dg','Pp2-dd','Pp2-da','Pp2-dl','Pp2-di','Pp2-pg','Pp2-pd','Pp2-pa','Pp2-pl','Pp2-pi','Pp2mdn','Pp2mpn',
              'Pp2fdn','Pp2fpn','Pp2ndn','Pp2npn','Pp3msn','Pp3msg','Pp3msg--y','Pp3msd','Pp3msd--y','Pp3msa','Pp3msa--y',
              'Pp3msa--b','Pp3msl','Pp3msi','Pp3mdn','Pp3mdg','Pp3mdg--y','Pp3mdd','Pp3mdd--y','Pp3mda','Pp3mda--y','Pp3mda--b',
              'Pp3mdl','Pp3mdi','Pp3mpn','Pp3mpg','Pp3mpg--y','Pp3mpd','Pp3mpd--y','Pp3mpa','Pp3mpa--y','Pp3mpa--b','Pp3mpl',
              'Pp3mpi','Pp3fsn','Pp3fsg','Pp3fsg--y','Pp3fsd','Pp3fsd--y','Pp3fsa','Pp3fsa--y','Pp3fsa--b','Pp3fsl','Pp3fsi',
              'Pp3fdn','Pp3fdg','Pp3fdg--y','Pp3fdd','Pp3fdd--y','Pp3fda','Pp3fda--y','Pp3fda--b','Pp3fdl','Pp3fdi','Pp3fpn',
              'Pp3fpg','Pp3fpg--y','Pp3fpd','Pp3fpd--y','Pp3fpa','Pp3fpa--y','Pp3fpa--b','Pp3fpl','Pp3fpi','Pp3nsn','Pp3nsg',
              'Pp3nsg--y','Pp3nsd','Pp3nsd--y','Pp3nsa','Pp3nsa--y','Pp3nsa--b','Pp3nsl','Pp3nsi','Pp3ndn','Pp3ndg','Pp3ndg--y',
              'Pp3ndd','Pp3ndd--y','Pp3nda','Pp3nda--y','Pp3nda--b','Pp3ndl','Pp3ndi','Pp3npn','Pp3npg','Pp3npg--y','Pp3npd',
              'Pp3npd--y','Pp3npa','Pp3npa--y','Pp3npa--b','Pp3npl','Pp3npi','Ps1msns','Ps1msnd','Ps1msnp','Ps1msgs','Ps1msgd',
              'Ps1msgp','Ps1msds','Ps1msdd','Ps1msdp','Ps1msas','Ps1msad','Ps1msap','Ps1msls','Ps1msld','Ps1mslp','Ps1msis',
              'Ps1msid','Ps1msip','Ps1mdns','Ps1mdnd','Ps1mdnp','Ps1mdgs','Ps1mdgd','Ps1mdgp','Ps1mdds','Ps1mddd','Ps1mddp',
              'Ps1mdas','Ps1mdad','Ps1mdap','Ps1mdls','Ps1mdld','Ps1mdlp','Ps1mdis','Ps1mdid','Ps1mdip','Ps1mpns','Ps1mpnd',
              'Ps1mpnp','Ps1mpgs','Ps1mpgd','Ps1mpgp','Ps1mpds','Ps1mpdd','Ps1mpdp','Ps1mpas','Ps1mpad','Ps1mpap','Ps1mpls',
              'Ps1mpld','Ps1mplp','Ps1mpis','Ps1mpid','Ps1mpip','Ps1fsns','Ps1fsnd','Ps1fsnp','Ps1fsgs','Ps1fsgd','Ps1fsgp',
              'Ps1fsds','Ps1fsdd','Ps1fsdp','Ps1fsas','Ps1fsad','Ps1fsap','Ps1fsls','Ps1fsld','Ps1fslp','Ps1fsis','Ps1fsid',
              'Ps1fsip','Ps1fdns','Ps1fdnd','Ps1fdnp','Ps1fdgs','Ps1fdgd','Ps1fdgp','Ps1fdds','Ps1fddd','Ps1fddp','Ps1fdas',
              'Ps1fdad','Ps1fdap','Ps1fdls','Ps1fdld','Ps1fdlp','Ps1fdis','Ps1fdid','Ps1fdip','Ps1fpns','Ps1fpnd','Ps1fpnp',
              'Ps1fpgs','Ps1fpgd','Ps1fpgp','Ps1fpds','Ps1fpdd','Ps1fpdp','Ps1fpas','Ps1fpad','Ps1fpap','Ps1fpls','Ps1fpld',
              'Ps1fplp','Ps1fpis','Ps1fpid','Ps1fpip','Ps1nsns','Ps1nsnd','Ps1nsnp','Ps1nsgs','Ps1nsgd','Ps1nsgp','Ps1nsds',
              'Ps1nsdd','Ps1nsdp','Ps1nsas','Ps1nsad','Ps1nsap','Ps1nsls','Ps1nsld','Ps1nslp','Ps1nsis','Ps1nsid','Ps1nsip',
              'Ps1ndns','Ps1ndnd','Ps1ndnp','Ps1ndgs','Ps1ndgd','Ps1ndgp','Ps1ndds','Ps1nddd','Ps1nddp','Ps1ndas','Ps1ndad',
              'Ps1ndap','Ps1ndls','Ps1ndld','Ps1ndlp','Ps1ndis','Ps1ndid','Ps1ndip','Ps1npns','Ps1npnd','Ps1npnp','Ps1npgs',
              'Ps1npgd','Ps1npgp','Ps1npds','Ps1npdd','Ps1npdp','Ps1npas','Ps1npad','Ps1npap','Ps1npls','Ps1npld','Ps1nplp',
              'Ps1npis','Ps1npid','Ps1npip','Ps2msns','Ps2msnd','Ps2msnp','Ps2msgs','Ps2msgd','Ps2msgp','Ps2msds','Ps2msdd',
              'Ps2msdp','Ps2msas','Ps2msad','Ps2msap','Ps2msls','Ps2msld','Ps2mslp','Ps2msis','Ps2msid','Ps2msip','Ps2mdns',
              'Ps2mdnd','Ps2mdnp','Ps2mdgs','Ps2mdgd','Ps2mdgp','Ps2mdds','Ps2mddd','Ps2mddp','Ps2mdas','Ps2mdad','Ps2mdap',
              'Ps2mdls','Ps2mdld','Ps2mdlp','Ps2mdis','Ps2mdid','Ps2mdip','Ps2mpns','Ps2mpnd','Ps2mpnp','Ps2mpgs','Ps2mpgd',
              'Ps2mpgp','Ps2mpds','Ps2mpdd','Ps2mpdp','Ps2mpas','Ps2mpad','Ps2mpap','Ps2mpls','Ps2mpld','Ps2mplp','Ps2mpis',
              'Ps2mpid','Ps2mpip','Ps2fsns','Ps2fsnd','Ps2fsnp','Ps2fsgs','Ps2fsgd','Ps2fsgp','Ps2fsds','Ps2fsdd','Ps2fsdp',
              'Ps2fsas','Ps2fsad','Ps2fsap','Ps2fsls','Ps2fsld','Ps2fslp','Ps2fsis','Ps2fsid','Ps2fsip','Ps2fdns','Ps2fdnd',
              'Ps2fdnp','Ps2fdgs','Ps2fdgd','Ps2fdgp','Ps2fdds','Ps2fddd','Ps2fddp','Ps2fdas','Ps2fdad','Ps2fdap','Ps2fdls',
              'Ps2fdld','Ps2fdlp','Ps2fdis','Ps2fdid','Ps2fdip','Ps2fpns','Ps2fpnd','Ps2fpnp','Ps2fpgs','Ps2fpgd','Ps2fpgp',
              'Ps2fpds','Ps2fpdd','Ps2fpdp','Ps2fpas','Ps2fpad','Ps2fpap','Ps2fpls','Ps2fpld','Ps2fplp','Ps2fpis','Ps2fpid',
              'Ps2fpip','Ps2nsns','Ps2nsnd','Ps2nsnp','Ps2nsgs','Ps2nsgd','Ps2nsgp','Ps2nsds','Ps2nsdd','Ps2nsdp','Ps2nsas',
              'Ps2nsad','Ps2nsap','Ps2nsls','Ps2nsld','Ps2nslp','Ps2nsis','Ps2nsid','Ps2nsip','Ps2ndns','Ps2ndnd','Ps2ndnp',
              'Ps2ndgs','Ps2ndgd','Ps2ndgp','Ps2ndds','Ps2nddd','Ps2nddp','Ps2ndas','Ps2ndad','Ps2ndap','Ps2ndls','Ps2ndld',
              'Ps2ndlp','Ps2ndis','Ps2ndid','Ps2ndip','Ps2npns','Ps2npnd','Ps2npnp','Ps2npgs','Ps2npgd','Ps2npgp','Ps2npds',
              'Ps2npdd','Ps2npdp','Ps2npas','Ps2npad','Ps2npap','Ps2npls','Ps2npld','Ps2nplp','Ps2npis','Ps2npid','Ps2npip',
              'Ps3msnsm','Ps3msnsf','Ps3msnsn','Ps3msnd','Ps3msnp','Ps3msgsm','Ps3msgsf','Ps3msgsn','Ps3msgd','Ps3msgp',
              'Ps3msdsm','Ps3msdsf','Ps3msdsn','Ps3msdd','Ps3msdp','Ps3msasm','Ps3msasf','Ps3msasn','Ps3msad','Ps3msap',
              'Ps3mslsm','Ps3mslsf','Ps3mslsn','Ps3msld','Ps3mslp','Ps3msism','Ps3msisf','Ps3msisn','Ps3msid','Ps3msip',
              'Ps3mdnsm','Ps3mdnsf','Ps3mdnsn','Ps3mdnd','Ps3mdnp','Ps3mdgsm','Ps3mdgsf','Ps3mdgsn','Ps3mdgd','Ps3mdgp',
              'Ps3mddsm','Ps3mddsf','Ps3mddsn','Ps3mddd','Ps3mddp','Ps3mdasm','Ps3mdasf','Ps3mdasn','Ps3mdad','Ps3mdap',
              'Ps3mdlsm','Ps3mdlsf','Ps3mdlsn','Ps3mdld','Ps3mdlp','Ps3mdism','Ps3mdisf','Ps3mdisn','Ps3mdid','Ps3mdip',
              'Ps3mpnsm','Ps3mpnsf','Ps3mpnsn','Ps3mpnd','Ps3mpnp','Ps3mpgsm','Ps3mpgsf','Ps3mpgsn','Ps3mpgd','Ps3mpgp',
              'Ps3mpdsm','Ps3mpdsf','Ps3mpdsn','Ps3mpdd','Ps3mpdp','Ps3mpasm','Ps3mpasf','Ps3mpasn','Ps3mpad','Ps3mpap',
              'Ps3mplsm','Ps3mplsf','Ps3mplsn','Ps3mpld','Ps3mplp','Ps3mpism','Ps3mpisf','Ps3mpisn','Ps3mpid','Ps3mpip',
              'Ps3fsnsm','Ps3fsnsf','Ps3fsnsn','Ps3fsnd','Ps3fsnp','Ps3fsgsm','Ps3fsgsf','Ps3fsgsn','Ps3fsgd','Ps3fsgp',
              'Ps3fsdsm','Ps3fsdsf','Ps3fsdsn','Ps3fsdd','Ps3fsdp','Ps3fsasm','Ps3fsasf','Ps3fsasn','Ps3fsad','Ps3fsap',
              'Ps3fslsm','Ps3fslsf','Ps3fslsn','Ps3fsld','Ps3fslp','Ps3fsism','Ps3fsisf','Ps3fsisn','Ps3fsid','Ps3fsip',
              'Ps3fdnsm','Ps3fdnsf','Ps3fdnsn','Ps3fdnd','Ps3fdnp','Ps3fdgsm','Ps3fdgsf','Ps3fdgsn','Ps3fdgd','Ps3fdgp',
              'Ps3fddsm','Ps3fddsf','Ps3fddsn','Ps3fddd','Ps3fddp','Ps3fdasm','Ps3fdasf','Ps3fdasn','Ps3fdad','Ps3fdap',
              'Ps3fdlsm','Ps3fdlsf','Ps3fdlsn','Ps3fdld','Ps3fdlp','Ps3fdism','Ps3fdisf','Ps3fdisn','Ps3fdid','Ps3fdip',
              'Ps3fpnsm','Ps3fpnsf','Ps3fpnsn','Ps3fpnd','Ps3fpnp','Ps3fpgsm','Ps3fpgsf','Ps3fpgsn','Ps3fpgd','Ps3fpgp',
              'Ps3fpdsm','Ps3fpdsf','Ps3fpdsn','Ps3fpdd','Ps3fpdp','Ps3fpasm','Ps3fpasf','Ps3fpasn','Ps3fpad','Ps3fpap',
              'Ps3fplsm','Ps3fplsf','Ps3fplsn','Ps3fpld','Ps3fplp','Ps3fpism','Ps3fpisf','Ps3fpisn','Ps3fpid','Ps3fpip',
              'Ps3nsnsm','Ps3nsnsf','Ps3nsnsn','Ps3nsnd','Ps3nsnp','Ps3nsgsm','Ps3nsgsf','Ps3nsgsn','Ps3nsgd','Ps3nsgp',
              'Ps3nsdsm','Ps3nsdsf','Ps3nsdsn','Ps3nsdd','Ps3nsdp','Ps3nsasm','Ps3nsasf','Ps3nsasn','Ps3nsad','Ps3nsap',
              'Ps3nslsm','Ps3nslsf','Ps3nslsn','Ps3nsld','Ps3nslp','Ps3nsism','Ps3nsisf','Ps3nsisn','Ps3nsid','Ps3nsip',
              'Ps3ndnsm','Ps3ndnsf','Ps3ndnsn','Ps3ndnd','Ps3ndnp','Ps3ndgsm','Ps3ndgsf','Ps3ndgsn','Ps3ndgd','Ps3ndgp',
              'Ps3nddsm','Ps3nddsf','Ps3nddsn','Ps3nddd','Ps3nddp','Ps3ndasm','Ps3ndasf','Ps3ndasn','Ps3ndad','Ps3ndap',
              'Ps3ndlsm','Ps3ndlsf','Ps3ndlsn','Ps3ndld','Ps3ndlp','Ps3ndism','Ps3ndisf','Ps3ndisn','Ps3ndid','Ps3ndip',
              'Ps3npnsm','Ps3npnsf','Ps3npnsn','Ps3npnd','Ps3npnp','Ps3npgsm','Ps3npgsf','Ps3npgsn','Ps3npgd','Ps3npgp',
              'Ps3npdsm','Ps3npdsf','Ps3npdsn','Ps3npdd','Ps3npdp','Ps3npasm','Ps3npasf','Ps3npasn','Ps3npad','Ps3npap',
              'Ps3nplsm','Ps3nplsf','Ps3nplsn','Ps3npld','Ps3nplp','Ps3npism','Ps3npisf','Ps3npisn','Ps3npid','Ps3npip',
              'Pd-msn','Pd-msg','Pd-msd','Pd-msa','Pd-msl','Pd-msi','Pd-mdn','Pd-mdg','Pd-mdd','Pd-mda','Pd-mdl','Pd-mdi',
              'Pd-mpn','Pd-mpg','Pd-mpd','Pd-mpa','Pd-mpl','Pd-mpi','Pd-fsn','Pd-fsg','Pd-fsd','Pd-fsa','Pd-fsl','Pd-fsi',
              'Pd-fdn','Pd-fdg','Pd-fdd','Pd-fda','Pd-fdl','Pd-fdi','Pd-fpn','Pd-fpg','Pd-fpd','Pd-fpa','Pd-fpl','Pd-fpi',
              'Pd-nsn','Pd-nsg','Pd-nsd','Pd-nsa','Pd-nsl','Pd-nsi','Pd-ndn','Pd-ndg','Pd-ndd','Pd-nda','Pd-ndl','Pd-ndi',
              'Pd-npn','Pd-npg','Pd-npd','Pd-npa','Pd-npl','Pd-npi','Pr','Pr----sm','Pr-msn','Pr-msg','Pr-msd','Pr-msa',
              'Pr-msl','Pr-msi','Pr-mdn','Pr-mdg','Pr-mdd','Pr-mda','Pr-mdl','Pr-mdi','Pr-mpn','Pr-mpg','Pr-mpd','Pr-mpa',
              'Pr-mpl','Pr-mpi','Pr-fsn','Pr-fsg','Pr-fsd','Pr-fsa','Pr-fsl','Pr-fsi','Pr-fdn','Pr-fdg','Pr-fdd','Pr-fda',
              'Pr-fdl','Pr-fdi','Pr-fpn','Pr-fpg','Pr-fpd','Pr-fpa','Pr-fpl','Pr-fpi','Pr-nsn','Pr-nsg','Pr-nsd','Pr-nsa',
              'Pr-nsl','Pr-nsi','Pr-ndn','Pr-ndg','Pr-ndd','Pr-nda','Pr-ndl','Pr-ndi','Pr-npn','Pr-npg','Pr-npd','Pr-npa',
              'Pr-npl','Pr-npi','Px------y','Px---g','Px---d','Px---d--y','Px---a','Px---a--b','Px---l','Px---i','Px-msn',
              'Px-msg','Px-msd','Px-msa','Px-msl','Px-msi','Px-mdn','Px-mdg','Px-mdd','Px-mda','Px-mdl','Px-mdi','Px-mpn',
              'Px-mpg','Px-mpd','Px-mpa','Px-mpl','Px-mpi','Px-fsn','Px-fsg','Px-fsd','Px-fsa','Px-fsl','Px-fsi','Px-fdn',
              'Px-fdg','Px-fdd','Px-fda','Px-fdl','Px-fdi','Px-fpn','Px-fpg','Px-fpd','Px-fpa','Px-fpl','Px-fpi','Px-nsn',
              'Px-nsg','Px-nsd','Px-nsa','Px-nsl','Px-nsi','Px-ndn','Px-ndg','Px-ndd','Px-nda','Px-ndl','Px-ndi','Px-npn',
              'Px-npg','Px-npd','Px-npa','Px-npl','Px-npi','Pg-msn','Pg-msg','Pg-msd','Pg-msa','Pg-msl','Pg-msi','Pg-mdn',
              'Pg-mdg','Pg-mdd','Pg-mda','Pg-mdl','Pg-mdi','Pg-mpn','Pg-mpg','Pg-mpd','Pg-mpa','Pg-mpl','Pg-mpi','Pg-fsn',
              'Pg-fsg','Pg-fsd','Pg-fsa','Pg-fsl','Pg-fsi','Pg-fdn','Pg-fdg','Pg-fdd','Pg-fda','Pg-fdl','Pg-fdi','Pg-fpn',
              'Pg-fpg','Pg-fpd','Pg-fpa','Pg-fpl','Pg-fpi','Pg-nsn','Pg-nsg','Pg-nsd','Pg-nsa','Pg-nsl','Pg-nsi','Pg-ndn',
              'Pg-ndg','Pg-ndd','Pg-nda','Pg-ndl','Pg-ndi','Pg-npn','Pg-npg','Pg-npd','Pg-npa','Pg-npl','Pg-npi','Pq----sm',
              'Pq----sf','Pq----sn','Pq----d','Pq----p','Pq-msn','Pq-msg','Pq-msd','Pq-msa','Pq-msl','Pq-msi','Pq-mdn',
              'Pq-mdg','Pq-mdd','Pq-mda','Pq-mdl','Pq-mdi','Pq-mpn','Pq-mpg','Pq-mpd','Pq-mpa','Pq-mpl','Pq-mpi','Pq-fsn',
              'Pq-fsg','Pq-fsd','Pq-fsa','Pq-fsl','Pq-fsi','Pq-fdn','Pq-fdg','Pq-fdd','Pq-fda','Pq-fdl','Pq-fdi','Pq-fpn',
              'Pq-fpg','Pq-fpd','Pq-fpa','Pq-fpl','Pq-fpi','Pq-nsn','Pq-nsg','Pq-nsd','Pq-nsa','Pq-nsl','Pq-nsi','Pq-ndn',
              'Pq-ndg','Pq-ndd','Pq-nda','Pq-ndl','Pq-ndi','Pq-npn','Pq-npg','Pq-npd','Pq-npa','Pq-npl','Pq-npi','Pi-msn',
              'Pi-msg','Pi-msd','Pi-msa','Pi-msl','Pi-msi','Pi-mdn','Pi-mdg','Pi-mdd','Pi-mda','Pi-mdl','Pi-mdi','Pi-mpn',
              'Pi-mpg','Pi-mpd','Pi-mpa','Pi-mpl','Pi-mpi','Pi-fsn','Pi-fsg','Pi-fsd','Pi-fsa','Pi-fsl','Pi-fsi','Pi-fdn',
              'Pi-fdg','Pi-fdd','Pi-fda','Pi-fdl','Pi-fdi','Pi-fpn','Pi-fpg','Pi-fpd','Pi-fpa','Pi-fpl','Pi-fpi','Pi-nsn',
              'Pi-nsg','Pi-nsd','Pi-nsa','Pi-nsl','Pi-nsi','Pi-ndn','Pi-ndg','Pi-ndd','Pi-nda','Pi-ndl','Pi-ndi','Pi-npn',
              'Pi-npg','Pi-npd','Pi-npa','Pi-npl','Pi-npi','Pz-msn','Pz-msg','Pz-msd','Pz-msa','Pz-msl','Pz-msi','Pz-mdn',
              'Pz-mdg','Pz-mdd','Pz-mda','Pz-mdl','Pz-mdi','Pz-mpn','Pz-mpg','Pz-mpd','Pz-mpa','Pz-mpl','Pz-mpi','Pz-fsn',
              'Pz-fsg','Pz-fsd','Pz-fsa','Pz-fsl','Pz-fsi','Pz-fdn','Pz-fdg','Pz-fdd','Pz-fda','Pz-fdl','Pz-fdi','Pz-fpn',
              'Pz-fpg','Pz-fpd','Pz-fpa','Pz-fpl','Pz-fpi','Pz-nsn','Pz-nsg','Pz-nsd','Pz-nsa','Pz-nsl','Pz-nsi','Pz-ndn',
              'Pz-ndg','Pz-ndd','Pz-nda','Pz-ndl','Pz-ndi','Pz-npn','Pz-npg','Pz-npd','Pz-npa','Pz-npl','Pz-npi','Mdc','Mdo',
              'Mrc','Mro','Mlc-pn','Mlc-pg','Mlc-pd','Mlc-pa','Mlc-pl','Mlc-pi','Mlcmdn','Mlcmdg','Mlcmdd','Mlcmda','Mlcmdl',
              'Mlcmdi','Mlcmpn','Mlcmpg','Mlcmpd','Mlcmpa','Mlcmpl','Mlcmpi','Mlcfdn','Mlcfdg','Mlcfdd','Mlcfda','Mlcfdl',
              'Mlcfdi','Mlcfpn','Mlcfpg','Mlcfpd','Mlcfpa','Mlcfpl','Mlcfpi','Mlcndn','Mlcndg','Mlcndd','Mlcnda','Mlcndl',
              'Mlcndi','Mlcnpn','Mlcnpg','Mlcnpd','Mlcnpa','Mlcnpl','Mlcnpi','Mlomsn','Mlomsg','Mlomsd','Mlomsa','Mlomsl',
              'Mlomsi','Mlomdn','Mlomdg','Mlomdd','Mlomda','Mlomdl','Mlomdi','Mlompn','Mlompg','Mlompd','Mlompa','Mlompl',
              'Mlompi','Mlofsn','Mlofsg','Mlofsd','Mlofsa','Mlofsl','Mlofsi','Mlofdn','Mlofdg','Mlofdd','Mlofda','Mlofdl',
              'Mlofdi','Mlofpn','Mlofpg','Mlofpd','Mlofpa','Mlofpl','Mlofpi','Mlonsn','Mlonsg','Mlonsd','Mlonsa','Mlonsl',
              'Mlonsi','Mlondn','Mlondg','Mlondd','Mlonda','Mlondl','Mlondi','Mlonpn','Mlonpg','Mlonpd','Mlonpa','Mlonpl',
              'Mlonpi','Mlpmsn','Mlpmsnn','Mlpmsny','Mlpmsg','Mlpmsd','Mlpmsa','Mlpmsan','Mlpmsay','Mlpmsl','Mlpmsi','Mlpmdn',
              'Mlpmdg','Mlpmdd','Mlpmda','Mlpmdl','Mlpmdi','Mlpmpn','Mlpmpg','Mlpmpd','Mlpmpa','Mlpmpl','Mlpmpi','Mlpfsn',
              'Mlpfsg','Mlpfsd','Mlpfsa','Mlpfsl','Mlpfsi','Mlpfdn','Mlpfdg','Mlpfdd','Mlpfda','Mlpfdl','Mlpfdi','Mlpfpn',
              'Mlpfpg','Mlpfpd','Mlpfpa','Mlpfpl','Mlpfpi','Mlpnsn','Mlpnsg','Mlpnsd','Mlpnsa','Mlpnsl','Mlpnsi','Mlpndn',
              'Mlpndg','Mlpndd','Mlpnda','Mlpndl','Mlpndi','Mlpnpn','Mlpnpg','Mlpnpd','Mlpnpa','Mlpnpl','Mlpnpi','Mlsmsnn',
              'Mlsmsny','Mlsmsg','Mlsmsd','Mlsmsa','Mlsmsan','Mlsmsay','Mlsmsl','Mlsmsi','Mlsmdn','Mlsmdg','Mlsmdd','Mlsmda',
              'Mlsmdl','Mlsmdi','Mlsmpn','Mlsmpg','Mlsmpd','Mlsmpa','Mlsmpl','Mlsmpi','Mlsfsn','Mlsfsg','Mlsfsd','Mlsfsa',
              'Mlsfsl','Mlsfsi','Mlsfdn','Mlsfdg','Mlsfdd','Mlsfda','Mlsfdl','Mlsfdi','Mlsfpn','Mlsfpg','Mlsfpd','Mlsfpa',
              'Mlsfpl','Mlsfpi','Mlsnsn','Mlsnsg','Mlsnsd','Mlsnsa','Mlsnsl','Mlsnsi','Mlsndn','Mlsndg','Mlsndd','Mlsnda',
              'Mlsndl','Mlsndi','Mlsnpn','Mlsnpg','Mlsnpd','Mlsnpa','Mlsnpl','Mlsnpi','Sn','Sg','Sd','Sa','Sl','Si','Cc',
              'Cs','Q','I','Y','X','Xf','Xt','Xp','Γ'] #1903


        

        
def loadData(dataset, lenghtOfWorld, numberOfTags):
    rawMatrix = ''
    tree = ET.iterparse(dataset)
    sentenceTagFirst = True
    first = True
    for _, el in tree:
        if '}' in el.tag:
            if el.tag.split('}', 1)[1] == 'w' or el.tag.split('}', 1)[1] == 's':
                if el.tag.split('}', 1)[1] == 'w':
                    if sentenceTagFirst:
                        sentenceTag = addEmptyTags(numberOfTags).astype(int)
                        sentenceTagFirst = False
                    processedTag = processTag(el.get('ana').split('msd:',1)[1])
                    #tagIndex = tagset.index(el.get('ana').split('msd:',1)[1])
                    enc = encodeWord(el.text,sentenceTag, lenghtOfWorld)
                    #rawRow = np.vstack([tagIndex,encodeWord(el.text,sentenceTag, lenghtOfWorld).T])
                    rawRow = np.hstack([processedTag,enc]) # 9characters of tag, 4*9tags+n (4)chars of word
                   
                    #sentenceTag = np.hstack(((sentenceTag)[:,9:],processedTag))  
                    sentenceTag = np.hstack(((sentenceTag)[:,351:],processedTag)) #Remove first tag - tag has 9 characters
                else:
                    sentenceTagFirst = True
                if first:
                    #rawMatrix = rawRow.T
                    rawMatrix = rawRow
                    first = False
                else:
                    #rawMatrix = np.vstack((rawMatrix,rawRow.T))
                    rawMatrix = np.vstack((rawMatrix,rawRow))
                    r, c = rawMatrix.shape
                    if r>=190000:  #STOP LOADING
                        break
    return rawMatrix

def addEmptyTags(numberOfTags):
    emptyTags = np.array(())
    for i in range(numberOfTags):
        et = processTag('Γ')
        if emptyTags.size == 0:
            emptyTags = et
        else:
            emptyTags = np.hstack((emptyTags,et))
    #return encodeChars(emptyTags)
    return emptyTags
 
def processTag(tag):

    totalLen = 9 # Tag has maximum of 9 characters
    if len(tag)<totalLen:
        numOfChars = totalLen-len(tag)
        tagConverted = numOfChars*'Λ'+tag
    else:
        tagConverted = tag
    a = oneHotEncodeTag(tagConverted)
    return oneHotEncodeTag(tagConverted)
#    return encodeChars(tagConverted) 
    #return tagConverted
    
def encodeWord(word, tag, lenghth):
    if len(word)<lenghth:
        numOfChars = lenghth-len(word)
        wordConverted = numOfChars*'λ'+word[::-1]
    else:
        wordShort = word[-lenghth:]
        wordConverted = wordShort[::-1]
    r = encodeChars(wordConverted).reshape((1,33*lenghth))
    return np.hstack((tag,r))
    #return np.hstack((tag,wordConverted))

def encodeChars(word):
    return oneHotEncode(np.array([list(char) for char in word]).T)
    
def oneHotEncodeTag(tag):
    numbers = ['b', 'g', 'a', 'p', 's', 'M', 'o', 'e', 'l', 'N', 'A', 'd', 'X', 'Q', 'Y', 'u', 'I', 'q', 'P', 'i', 'y', 'n', 'f', 'r', 'R', 'x', 'c', 'z', '1', 'V', 'm', 'S', '2', 'C', '3', 't', '-','Λ','Γ'] #39
    encoded = np.array([])  
    numOfChars = len(tag)
    for c in range(numOfChars):
        ch = tag[c]
        num = numbers.index(ch)
        encoded = np.hstack((encoded,num)).astype(int)
    b = np.zeros((numOfChars,39))
    b[np.arange(numOfChars),encoded] = 1
    b = b.reshape(1,numOfChars*39)
    return b
    
def oneHotEncode(chars):
    numbers = ['a','b','c','č','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','š','t','u','v','w','x','y','z','ž','λ','Λ','Γ'] #32
    numOfChars = chars.size
    encoded = np.array([])
    chars = chars[0]
    for c in range(numOfChars):
        ch = chars.T[c].lower()
        if ch in numbers:
            num = numbers.index(ch)
        else:
            num = 32
        encoded = np.hstack((encoded,num)).astype(int)
    b = np.zeros((numOfChars,33))
    b[np.arange(numOfChars), encoded] = 1        
    return b


## TEST BAYES START ###########################

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]
 
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated
 
def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
#    avg = mean(numbers)
#    if (len(numbers)-1) == 0:
#        variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers))
#    else:
#        variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return np.std(numbers)#math.sqrt(variance)
 
 

def saveData(data):
    f = open('POS_data_4_1HotVector_full.pkl','wb')
    pickle.dump(data,f)
    f.close
    print("Saved!")
    
def loadDataFromFile():
    f = open('POS_data_4_1HotVector_full.pkl','rb')
    data = pickle.load(f)
    return data
	
	
	

def test_spark_mlp():
    data = loadDataFromFile()
    np.random.shuffle(data)
    numRows, numCols = data.shape
    trainIndex = int(np.floor(numRows/10))*4
    xTrain = data[:trainIndex,351:]
    yTrain = data[:trainIndex,:351]
    xTest = data[trainIndex:,351:]
    yTest = data[trainIndex:,:351]
  
    mlp = MLPClassifier(hidden_layer_sizes=(1536,2000,351))
    print("...Training...")
    mlp.fit(xTrain,yTrain)
    print("...Testing...")
    predictions = mlp.predict(xTest)
    print(classification_report(yTest,predictions))

    
def test_spark_bayes():
    dataset='C:/Users/rlukas/Desktop/TrainingCorpus_ssj500kv1_4-EN/ssj500k-en.xml'
    #data = loadData(dataset, 4, 4)
    #saveData(data)
    data = loadDataFromFile()
    np.random.shuffle(data)
    numRows, numCols = data.shape
    trainIndex = int(np.floor(numRows/10))*4
    xTrain = data[:trainIndex,351:]
    yTrain = data[:trainIndex,:351]
    xTest = data[trainIndex:,351:]
    yTest = data[trainIndex:,:351]
    training = np.hstack((xTrain,yTrain))
    test = np.hstack((xTest,yTest))
    xdata = data[:,:-351]
    ydata = data[:,-351]
    lbin = LabelBinarizer()
    print("LabelBinarizer")
    for k in range(np.size(xdata,1)):
        if k==0:
            xdata_ml = lbin.fit_transform(xdata[:,k])
        else:
            xdata_ml = np.hstack((xdata_ml,lbin.fit_transform(xdata[:,k])))
    print("xdata_ml")
    ydata_ml = lbin.fit_transform(ydata)
    
    allIDX = np.arange(numRows)
    random.shuffle(allIDX)
    holdout_number = (numRows/10)*4
    testIDX = allIDX[0:holdout_number]
    trainIDX = allIDX[holdout_number:]

    xtest = xdata_ml[testIDX,:]
    xtrain = xdata_ml[trainIDX,:]
    ytest = ydata[testIDX]
    ytrain = ydata[trainIDX]
    #mnb = naive_bayes.GaussianNB()
    mnb = naive_bayes.MultinomialNB()
    mnb.fit(xtrain,ytrain)
    print("Classification accuracy of MultinomialNB = ", mnb.score(xtest,ytest))
    
    



    


test_spark_mlp()
#test_spark_bayes()
