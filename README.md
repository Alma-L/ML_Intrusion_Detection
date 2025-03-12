# University of Prishtina “Hasan Prishtina” <img src="https://upload.wikimedia.org/wikipedia/commons/e/e1/University_of_Prishtina_logo.svg" width="100" align="right">

*Faculty of Electrical and Computer Engineering*  
**Level:** Master  
**Course:** Machine Learning  
**Project Title:** Cybersecurity Intrusion Detection Dataset  

**Team Members:**  
- Alma Latifi  
- Endrit Balaj  
- Rinesa Bislimi 

---
The Cybersecurity Intrusion Detection Dataset contains network traffic data designed for detecting cyber threats and intrusions. It includes various network features such as packet headers, protocol types, and traffic patterns, enabling the training and evaluation of machine learning models for intrusion detection. 

---
## Data Dictionary (Columns Explanation)

| **Column Name**  | **Description** |
|---------------|--------------|
| **Duration** | Total duration of the session in seconds. |
| **Protocol** | Network protocol used (e.g., TCP, UDP, ICMP). |
| **SourceIP** | IP address of the sender. |
| **DestinationIP** | IP address of the receiver. |
| **SourcePort** | Port number of the sender. |
| **DestinationPort** | Port number of the receiver. |
| **PacketCount** | Number of packets exchanged in the session. |
| **ByteCount** | Total number of bytes exchanged. |
| **Label** | Classification label indicating normal or attack activity. |
| **session_id** | Unique identifier for each session (e.g., SID_00001). |
| **network_packet_size** | Size of network packets in bytes. |
| **protocol_type** | Communication protocol used (e.g., TCP, UDP, ICMP). |
| **login_attempts** | Number of login attempts during the session. |
| **session_duration** | Length of the session in seconds. |
| **encryption_used** | Type of encryption used (AES, DES, or None). |
| **ip_reputation_score** | Score between 0 and 1 indicating how suspicious the IP is. |
| **failed_logins** | Number of failed login attempts. |
| **browser_type** | Browser used for the session (e.g., Edge, Firefox). |
| **unusual_time_access** | Binary flag (0 or 1) indicating unusual access time. |
| **attack_detected** | Target variable: 1 means an attack was detected, 0 means normal activity. |

---

## Example Data

```plaintext
| session_id  | network_packet_size | protocol_type | login_attempts | session_duration | encryption_used | ip_reputation_score | failed_logins | browser_type | unusual_time_access | attack_detected |
|------------|----------------------|---------------|----------------|------------------|-----------------|----------------------|--------------|-------------|----------------------|-----------------|
| SID_00001  | 512                  | TCP           | 2              | 360.5            | AES             | 0.75                 | 1            | Chrome      | 0                    | 1               |
| SID_00002  | 204                  | UDP           | 0              | 120.8            | None            | 0.20                 | 0            | Firefox     | 1                    | 0               |
```
---


