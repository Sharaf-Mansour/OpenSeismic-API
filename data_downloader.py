import seisbench.data as sbd
import seisbench.util as sbu
from obspy.clients.fdsn import Client
import numpy as np
from obspy.clients.fdsn.header import FDSNException
from pathlib import Path
import requests
from obspy.core.event import read_events
  

def download_event_data(startdate, enddate):
    link = f"https://service.iris.edu/irisws/mars-event/1/query?starttime={startdate}T00:00:00&endtime={enddate}T00:00:00&magnitudetype=M2.4,MFB,MbS,MbP,MWspec,MW&eventtype=2.4Hz,HF,VF&locationquality=A,B,C&includeallorigins=false&includeallmagnitudes=false&includearrivals=true&version=14&orderby=time-asc&format=xml&nodata=204"
    r = requests.get(link)
    with open("events.xml", 'w') as f:
        f.write(r.text)
    catalog = read_events("events.xml")
    return catalog

def get_event_params(event):
    origin = event.preferred_origin()
    mag = event.preferred_magnitude()
    source_id = str(event.resource_id)
    event_params={
        'source_id': source_id,
        'latitude': origin['latitude'] if origin['latitude'] is not None else None,
        'latitude_errors':origin['latitude_errors']['uncertainty'] if origin['latitude_errors'] is not None else None,
        'longitude': origin['longitude'] if origin['longitude'] is not None else None,
        'longitude_errors': origin['longitude_errors']['uncertainty'] if origin['longitude_errors'] is not None else None,
        'depth': origin['depth'] if origin['depth']  is not None else None,
        'depth_errors': origin['depth_errors']['uncertainty'] if origin['depth_errors'] is not None else None,
        'depth_type':origin['depth_type'] if origin['depth_type']  is not None else None,
        'origin_evaluation_mode': origin['evaluation_mode'] if origin['evaluation_mode'] is not None else None,
        'origievaluation_status': origin['evaluation_status'] if origin['evaluation_status'] is not None else None,
        'time': origin['time'] if origin['time']  is not None else None,
        'region': origin['region'] if origin['region'] is not None else None,
      }
    if mag is not None:
        event_params['mag'] =  mag['mag'] if mag['mag'] is not None else None
        event_params['mag_errors'] =  mag['mag_errors']['uncertainty'] if mag['mag_errors'] is not None else None
        event_params['magnitude_type'] = mag['magnitude_type'] if mag['magnitude_type'] is not None else None
        event_params['creation_info'] =  mag['creation_info'] if mag['creation_info'] is not None else None
        event_params['evaluation_mode'] = mag['evaluation_mode'] if mag['evaluation_mode'] is not None else None
        event_params['evaluation_status'] = mag['evaluation_status'] if mag['evaluation_status'] is not None else None

# if origin.time < "2022-01-20":
#   split = 'train'
# elif origin.time<'2022-01-23':
#   split = 'val'
# else:
#   split = 'test'

# event_params["split"] = split

    return event_params


def get_trace_params(pick):
  net = pick.waveform_id.network_code
  sta = pick.waveform_id.station_code

  trace_params={
      'station_network_code':net,
      "station_code":sta,
      "trace_channel":pick.waveform_id.channel_code[:2],
      'station_location_code': pick.waveform_id.location_code,
  }
  return trace_params

def get_waveforms(start_time,end_time , trace_params, time_before=60, time_after=60 ):
    t_start , t_end = start_time- time_before , end_time + time_after
    try:
        client = Client('IRIS')
        waveforms = client.get_waveforms(
            network=trace_params["station_network_code"],
            station=trace_params["station_code"],
            location=trace_params['station_location_code'],
            channel=f"{trace_params['trace_channel']}*",
            starttime=t_start,
            endtime=t_end,
        )
    except FDSNException:
        # Return empty stream
        return []
         

    return waveforms
def get_picks (event):
    list_of_picks=  ['P', 'S','p','s','Pg','Sg','pP','P1','Pn',"PmP","pwP",'pwPm', 'S1','SmS','Sn']
    picks = [x for x in event.picks if x.phase_hint in list_of_picks]
    if len(picks) == 0:
        return []
    picks = [[picks[i],picks[i+1]] for i in range (0 , len(picks),2) ]

    return sorted( picks[0] , key = lambda x: x.time)
def switch_UVW_to_ZNE(stream):
    conv = {'U': 'Z', 'V':'N' , "W": "E"}
    for i in stream:
        i.stats.channel = i.stats.channel[:-1]+ conv[i.stats.channel[-1]]
    return stream
def switch_ZNE_to_UVW(stream):
    conv = {'Z': 'U', 'N':'V' , "E": "W"}
    for i in stream:
        i.stats.channel = i.stats.channel[:-1]+ conv[i.stats.channel[-1]]
    return stream

def get_data(waveforms):
    data = [i.data for i in waveforms]
    min_len = min([len(x) for x in data])
    data = np.array([i[:min_len]for i in data ])
    return data


def write_data(startdate,enddate):
    catalog = download_event_data(startdate, enddate)
    base_path = Path('data_v2_denoised')
    metadata_path = base_path/"metadata.csv"
    waveforms_path = base_path/'waveforms.hdf5'
    with sbd.WaveformDataWriter(metadata_path, waveforms_path) as writer:
        # Define data format
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }

        for event in catalog:
            event_params = get_event_params(event)
            p = sorted([x.time for x in event.picks] )
            start_time , end_time = p[0], p[-1]
            trace_params = get_trace_params(event.picks[0])
            waveforms = get_waveforms( start_time , end_time , trace_params , time_before = 0 ,
                                    time_after=0 ,)
            if len(waveforms) == 0:
                # No waveform data available
                continue

            sampling_rate = waveforms[0].stats.sampling_rate
            # Check that the traces have the same sampling rate
            assert all(trace.stats.sampling_rate == sampling_rate for trace in waveforms)
            waveforms = switch_UVW_to_ZNE(waveforms)
            actual_t_start, _ , _ = sbu.stream_to_array(
                    waveforms,
                    component_order="ZNE",
                )
            data = get_data(waveforms)
            trace_params["trace_sampling_rate_hz"] = 100
            trace_params["trace_start_time"] = str(actual_t_start)
            for pick in event.picks :
                trace_params[f"trace_{pick.phase_hint}_time"] = pick.time
                sample = (pick.time - actual_t_start) * sampling_rate
                trace_params[f"trace_{pick.phase_hint}_arrival_sample"] = int(sample)

                writer.add_trace({**event_params, **trace_params}, data)